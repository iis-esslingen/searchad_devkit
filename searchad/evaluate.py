import argparse
import csv
import math
import os
from collections import defaultdict
from datetime import datetime

import torch

from searchad.config.config import (
    MIN_BOX_AREA,
    SEARCHAD_CATEGORIES,
    SEARCHAD_LABELS,
    SEARCHAD_TRAIN_DATASETS,
    SEARCHAD_VAL_DATASETS,
)
from searchad.utils.checks import (
    check_ground_truth_coverage,
    check_prediction_dataset_prefixes,
    check_prediction_labels,
)
from searchad.utils.io import load_json, save_json
from searchad.utils.metrics import (
    calculate_category_averages,
    mean_average_precision,
    mean_rprecision,
    precision_at_k,
)


def _check_predictions(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    searchad_datasets: list[str],
) -> None:
    """Runs sanity checks on the loaded predictions and prints a summary."""
    print("\n--- Performing Sanity Checks on Predictions File ---")
    failed_count = 0

    failed_count += check_prediction_labels(predictions, SEARCHAD_LABELS)
    failed_count += check_prediction_dataset_prefixes(predictions, searchad_datasets)
    failed_count += check_ground_truth_coverage(predictions, ground_truth)

    if failed_count > 0:
        print(f"--- Predictions Sanity Checks Completed with {failed_count} issue(s) found ---")
    else:
        print("--- All Predictions Sanity Checks Passed ---")


def _evaluate_labels(
    imagelist: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    ignore_list: dict[str, list[str]],
) -> tuple[dict, float, float, dict[str, float]]:
    """
    Internal helper to evaluate individual labels and calculate overall scores.
    """
    results_meta_data = {}

    # Pre-process ignore_list and ground_truth into sets for O(1) average lookup
    processed_ignore_list = {label: set(paths) for label, paths in ignore_list.items()}
    processed_ground_truth = {label: set(paths) for label, paths in ground_truth.items()}

    all_map_scores = []
    all_mrp_scores = []
    all_pk_scores = defaultdict(list)  # To store P@k scores for aggregation

    k_values = [5, 25, 100]

    for label, imagepaths in imagelist.items():
        searchad_label = label
        match_indices_for_label = []

        current_ignore_paths_set = processed_ignore_list.get(searchad_label, set())
        current_gt_imagepaths_set = processed_ground_truth.get(searchad_label, set())

        # Filter out ignored images and create a mapping for efficient lookup
        filtered_imagepaths = []
        path_to_filtered_index = {}
        for path in imagepaths:
            if path not in current_ignore_paths_set:
                path_to_filtered_index[path] = len(filtered_imagepaths)
                filtered_imagepaths.append(path)

        matched = 0
        for gt_imagepath in current_gt_imagepaths_set:
            if gt_imagepath in path_to_filtered_index:
                index = path_to_filtered_index[gt_imagepath]
                match_indices_for_label.append(index)
                matched += 1

        match_indices_for_label.sort()

        results_meta_data[searchad_label] = {
            "total_predicted_images": len(imagepaths),
            "total_relevant_images_after_ignore": len(current_gt_imagepaths_set),
            "ignored_images_in_prediction": len(current_ignore_paths_set.intersection(set(imagepaths))),
            "num_matches": matched,
            "num_filtered_predictions": len(filtered_imagepaths),
        }

        top_k_predictions = len(filtered_imagepaths)

        # If there are no filtered predictions, MAP, R-Precision, and P@k are undefined or 0
        if top_k_predictions == 0:
            map_score = torch.tensor(float("nan"))
            mrp = torch.tensor(float("nan"))
            pk_scores = {f"P@{k_val}": torch.tensor(float("nan")) for k_val in k_values}
        else:
            targets = torch.zeros(top_k_predictions)
            for match_idx in match_indices_for_label:
                if match_idx < top_k_predictions:
                    targets[match_idx] = 1.0

            map_score = mean_average_precision(targets, top_k_predictions)
            mrp = mean_rprecision(targets)

            pk_scores = {}
            for k_val in k_values:
                # precision_at_k expects k to be less than or equal to the number of predictions
                # If top_k_predictions is less than k_val, P@k is calculated over available predictions
                pk_scores[f"P@{k_val}"] = precision_at_k(targets, k_val)

        results_meta_data[searchad_label]["MAP"] = map_score.item()
        results_meta_data[searchad_label]["R-Precision"] = mrp.item()
        for k_val in k_values:
            results_meta_data[searchad_label][f"P@{k_val}"] = pk_scores[f"P@{k_val}"].item()

        if not math.isnan(map_score.item()):
            all_map_scores.append(map_score.item())
        if not math.isnan(mrp.item()):
            all_mrp_scores.append(mrp.item())
        for k_val in k_values:
            pk_score = pk_scores[f"P@{k_val}"].item()
            if not math.isnan(pk_score):
                all_pk_scores[f"P@{k_val}"].append(pk_score)

    # Calculate overall MAP and MRP from collected valid scores
    overall_map = sum(all_map_scores) / len(all_map_scores) if all_map_scores else 0.0
    overall_mrp = sum(all_mrp_scores) / len(all_mrp_scores) if all_mrp_scores else 0.0

    overall_pk_results = {}
    for k_val in k_values:
        metric_name = f"P@{k_val}"
        scores_list = all_pk_scores[metric_name]
        overall_pk_results[metric_name] = sum(scores_list) / len(scores_list) if scores_list else 0.0

    return results_meta_data, overall_map, overall_mrp, overall_pk_results


def _load_evaluation_data(predictions_file: str, searchad_dir: str, split: str) -> tuple[dict, dict, dict]:
    """Loads prediction, ground truth, and ignore list data from JSON files."""
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    gt_file_path = os.path.join(searchad_dir, f"searchad_{split}_image_level_annotations_min_box_{MIN_BOX_AREA}.json")
    ignore_file_path = os.path.join(
        searchad_dir, f"searchad_{split}_image_level_ignore_images_min_box_{MIN_BOX_AREA}.json"
    )

    if not os.path.exists(gt_file_path):
        raise FileNotFoundError(f"Ground truth annotations file not found: {gt_file_path}")
    if not os.path.exists(ignore_file_path):
        raise FileNotFoundError(f"Ignore list file not found: {ignore_file_path}")

    predictions = load_json(predictions_file)
    ground_truth = load_json(gt_file_path)
    ignore_list = load_json(ignore_file_path)
    return predictions, ground_truth, ignore_list


def _format_metric_value(value: float) -> str:
    """Formats a float metric value or returns 'nan' if it's NaN."""
    return f"{value:.5f}" if not math.isnan(value) else "nan"


def _print_results_table(
    results_meta_data: dict,
    overall_map_score: float,
    overall_mrp_score: float,
    map_category_averages: dict,
    rp_category_averages: dict,
    overall_pk_results: dict[str, float],
    pk_category_averages: dict[str, dict[str, float | None]],
):
    """Prints the evaluation results to the console in a formatted table."""
    print("\n--- Evaluation Results ---")

    # Determine column widths for consistent formatting
    all_labels = (
        list(results_meta_data.keys())
        + list(map_category_averages.keys())
        + ["Overall", "SearchAD Classes", "Category"]
    )
    max_label_len = max(len(label) for label in all_labels) if all_labels else 0

    # Adjusted value_display_width to accommodate "Mean R-Precision"
    value_display_width = 14  # Increased from 10 to fit "Mean R-Precision"
    col1_width = max_label_len + 2
    col2_width = value_display_width + 2
    col3_width = value_display_width + 2
    col_pk_width = value_display_width + 2  # For P@k metrics

    header_map_text = "Mean AP"
    header_mrp_text = "Mean R-Precision"
    header_pk5_text = "P@5"
    header_pk25_text = "P@25"
    header_pk100_text = "P@100"

    # Print header for individual classes
    print(
        f"{'SearchAD Classes':<{col1_width}} | {header_map_text:^{col2_width}} | {header_mrp_text:^{col3_width}} | {header_pk5_text:^{col_pk_width}} | {header_pk25_text:^{col_pk_width}} | {header_pk100_text:^{col_pk_width}}"
    )
    print(
        f"{'-' * col1_width}-+-{'-' * col2_width}-+-{'-' * col3_width}-+-{'-' * col_pk_width}-+-{'-' * col_pk_width}-+-{'-' * col_pk_width}"
    )

    # Print individual class results
    for label in sorted(results_meta_data.keys()):
        map_val = results_meta_data[label]["MAP"]
        mrp_val = results_meta_data[label]["R-Precision"]
        pk5_val = results_meta_data[label]["P@5"]
        pk25_val = results_meta_data[label]["P@25"]
        pk100_val = results_meta_data[label]["P@100"]
        print(
            f"{label:<{col1_width}} | {_format_metric_value(map_val):^{col2_width}} | {_format_metric_value(mrp_val):^{col3_width}} | {_format_metric_value(pk5_val):^{col_pk_width}} | {_format_metric_value(pk25_val):^{col_pk_width}} | {_format_metric_value(pk100_val):^{col_pk_width}}"
        )

    # Print separator before overall results
    print(
        f"{'-' * col1_width}-+-{'-' * col2_width}-+-{'-' * col3_width}-+-{'-' * col_pk_width}-+-{'-' * col_pk_width}-+-{'-' * col_pk_width}"
    )

    # Print overall results
    print(
        f"{'Overall':<{col1_width}} | {_format_metric_value(overall_map_score):^{col2_width}} | {_format_metric_value(overall_mrp_score):^{col3_width}} | {_format_metric_value(overall_pk_results['P@5']):^{col_pk_width}} | {_format_metric_value(overall_pk_results['P@25']):^{col_pk_width}} | {_format_metric_value(overall_pk_results['P@100']):^{col_pk_width}}"
    )

    # Add Category-wise Averages to console output
    total_width = (
        col1_width + col2_width + col3_width + (3 * col_pk_width) + 15
    )  # 15 for the 5 separators × 3 chars each (" | " / "-+-")
    print(f"\n{'-' * total_width}")
    print(f"{'Category-wise Averages':^{total_width}}")
    print(f"{'-' * total_width}")

    print(
        f"{'Category':<{col1_width}} | {header_map_text:^{col2_width}} | {header_mrp_text:^{col3_width}} | {header_pk5_text:^{col_pk_width}} | {header_pk25_text:^{col_pk_width}} | {header_pk100_text:^{col_pk_width}}"
    )
    print(
        f"{'-' * col1_width}-+-{'-' * col2_width}-+-{'-' * col3_width}-+-{'-' * col_pk_width}-+-{'-' * col_pk_width}-+-{'-' * col_pk_width}"
    )

    for category in sorted(map_category_averages.keys()):
        map_val = map_category_averages[category]
        mrp_val = rp_category_averages[category]
        pk5_val = pk_category_averages["P@5"][category]
        pk25_val = pk_category_averages["P@25"][category]
        pk100_val = pk_category_averages["P@100"][category]

        map_str = f"{map_val:.5f}" if isinstance(map_val, (int, float)) else "N/A"
        mrp_str = f"{mrp_val:.5f}" if isinstance(mrp_val, (int, float)) else "N/A"
        pk5_str = f"{pk5_val:.5f}" if isinstance(pk5_val, (int, float)) else "N/A"
        pk25_str = f"{pk25_val:.5f}" if isinstance(pk25_val, (int, float)) else "N/A"
        pk100_str = f"{pk100_val:.5f}" if isinstance(pk100_val, (int, float)) else "N/A"

        print(
            f"{category:<{col1_width}} | {map_str:^{col2_width}} | {mrp_str:^{col3_width}} | {pk5_str:^{col_pk_width}} | {pk25_str:^{col_pk_width}} | {pk100_str:^{col_pk_width}}"
        )


def _save_results_to_csv(
    results_meta_data: dict,
    overall_map_score: float,
    overall_mrp_score: float,
    map_category_averages: dict,
    rp_category_averages: dict,
    overall_pk_results: dict[str, float],
    pk_category_averages: dict[str, dict[str, float | None]],
    scores_output_dir: str,
    predictions_file: str,
):
    """Saves the evaluation results to a CSV file."""
    if not os.path.exists(scores_output_dir):
        os.makedirs(scores_output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = os.path.splitext(os.path.basename(predictions_file))[0].lower()
    output_filename = f"{timestamp}_eval_{basename}.csv"
    output_filepath = os.path.join(scores_output_dir, output_filename)

    with open(output_filepath, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            [
                "SearchAD Classes",
                "Mean AP [%]",
                "Mean R-Precision [%]",
                "P@5 [%]",
                "P@25 [%]",
                "P@100 [%]",
            ]
        )
        for label in sorted(results_meta_data.keys()):
            map_val = results_meta_data[label]["MAP"]
            mrp_val = results_meta_data[label]["R-Precision"]
            pk5_val = results_meta_data[label]["P@5"]
            pk25_val = results_meta_data[label]["P@25"]
            pk100_val = results_meta_data[label]["P@100"]
            csv_writer.writerow(
                [
                    label,
                    f"{map_val * 100:.2f}" if not math.isnan(map_val) else "nan",
                    f"{mrp_val * 100:.2f}" if not math.isnan(mrp_val) else "nan",
                    f"{pk5_val * 100:.2f}" if not math.isnan(pk5_val) else "nan",
                    f"{pk25_val * 100:.2f}" if not math.isnan(pk25_val) else "nan",
                    f"{pk100_val * 100:.2f}" if not math.isnan(pk100_val) else "nan",
                ]
            )
        csv_writer.writerow(
            [
                "Overall",
                f"{overall_map_score * 100:.2f}" if not math.isnan(overall_map_score) else "nan",
                f"{overall_mrp_score * 100:.2f}" if not math.isnan(overall_mrp_score) else "nan",
                f"{overall_pk_results['P@5'] * 100:.2f}" if not math.isnan(overall_pk_results["P@5"]) else "nan",
                f"{overall_pk_results['P@25'] * 100:.2f}" if not math.isnan(overall_pk_results["P@25"]) else "nan",
                f"{overall_pk_results['P@100'] * 100:.2f}" if not math.isnan(overall_pk_results["P@100"]) else "nan",
            ]
        )

        csv_writer.writerow([])
        csv_writer.writerow(["Category-wise Averages", "", "", "", "", ""])
        csv_writer.writerow(["Category", "Mean AP [%]", "Mean R-Precision [%]", "P@5 [%]", "P@25 [%]", "P@100 [%]"])
        for category in sorted(map_category_averages.keys()):
            map_val = map_category_averages[category]
            mrp_val = rp_category_averages[category]
            pk5_val = pk_category_averages["P@5"][category]
            pk25_val = pk_category_averages["P@25"][category]
            pk100_val = pk_category_averages["P@100"][category]

            map_str = f"{map_val * 100:.2f}" if isinstance(map_val, (int, float)) and not math.isnan(map_val) else "N/A"
            mrp_str = f"{mrp_val * 100:.2f}" if isinstance(mrp_val, (int, float)) and not math.isnan(mrp_val) else "N/A"
            pk5_str = f"{pk5_val * 100:.2f}" if isinstance(pk5_val, (int, float)) and not math.isnan(pk5_val) else "N/A"
            pk25_str = (
                f"{pk25_val * 100:.2f}" if isinstance(pk25_val, (int, float)) and not math.isnan(pk25_val) else "N/A"
            )
            pk100_str = (
                f"{pk100_val * 100:.2f}" if isinstance(pk100_val, (int, float)) and not math.isnan(pk100_val) else "N/A"
            )

            csv_writer.writerow([category, map_str, mrp_str, pk5_str, pk25_str, pk100_str])

    print("\n--- Evaluation Results Scores saved to CSV ---")
    print(f"CSV file path: {output_filepath}")


def _save_results_to_json(
    results_meta_data: dict,
    overall_map_score: float,
    overall_mrp_score: float,
    map_category_averages: dict,
    rp_category_averages: dict,
    overall_pk_results: dict[str, float],
    pk_category_averages: dict[str, dict[str, float | None]],
    scores_output_dir: str,
    predictions_file: str,
):
    """Saves the evaluation results to a JSON file."""
    if not os.path.exists(scores_output_dir):
        os.makedirs(scores_output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = os.path.splitext(os.path.basename(predictions_file))[0].lower()
    output_filename = f"{timestamp}_eval_{basename}.json"
    output_filepath = os.path.join(scores_output_dir, output_filename)

    # Prepare data for JSON output
    json_output = {
        "overall_scores": {
            "MAP": overall_map_score if not math.isnan(overall_map_score) else None,
            "R-Precision": overall_mrp_score if not math.isnan(overall_mrp_score) else None,
            "P@5": overall_pk_results["P@5"] if not math.isnan(overall_pk_results["P@5"]) else None,
            "P@25": overall_pk_results["P@25"] if not math.isnan(overall_pk_results["P@25"]) else None,
            "P@100": overall_pk_results["P@100"] if not math.isnan(overall_pk_results["P@100"]) else None,
        },
        "category_averages": {
            "MAP": {
                k: (v if isinstance(v, (int, float)) and not math.isnan(v) else None)
                for k, v in map_category_averages.items()
            },
            "R-Precision": {
                k: (v if isinstance(v, (int, float)) and not math.isnan(v) else None)
                for k, v in rp_category_averages.items()
            },
            "P@5": {
                k: (v if isinstance(v, (int, float)) and not math.isnan(v) else None)
                for k, v in pk_category_averages["P@5"].items()
            },
            "P@25": {
                k: (v if isinstance(v, (int, float)) and not math.isnan(v) else None)
                for k, v in pk_category_averages["P@25"].items()
            },
            "P@100": {
                k: (v if isinstance(v, (int, float)) and not math.isnan(v) else None)
                for k, v in pk_category_averages["P@100"].items()
            },
        },
        "class_wise_results": {},
    }

    # Convert NaN values in class_wise_results to None for JSON compatibility
    for label, metrics in results_meta_data.items():
        json_output["class_wise_results"][label] = {
            k: (v if isinstance(v, (int, float)) and not math.isnan(v) else None) for k, v in metrics.items()
        }

    save_json(json_output, output_filepath)

    print("\n--- Evaluation Results Scores saved to JSON ---")
    print(f"JSON file path: {output_filepath}")


def evaluate(
    predictions_file: str,
    split: str,
    searchad_dir: str,
    scores_output_dir: str,
):
    """
    Runs the full evaluation process for SearchAD.

    Args:
        predictions_file (str): Path to the JSON file containing prediction results.
        split (str): The dataset split to evaluate ("train" or "val").
        searchad_dir (str): Path to the folder containing SearchAD annotation files
                              (ground truth and ignore lists).
        scores_output_dir (str): Path to the folder where evaluation scores (CSV, JSON)
                               should be saved.
    """
    print(f"Starting evaluation for split '{split}'...")

    if split not in ("train", "val"):
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'val'.")

    print(f"Loading predictions from: {predictions_file}")
    print(f"Loading ground truth and ignore lists from: {searchad_dir}")

    predictions, ground_truth, ignore_list = _load_evaluation_data(predictions_file, searchad_dir, split)
    searchad_datasets = SEARCHAD_TRAIN_DATASETS if split == "train" else SEARCHAD_VAL_DATASETS
    _check_predictions(predictions, ground_truth, searchad_datasets)

    results_meta_data, overall_map_score, overall_mrp_score, overall_pk_results = _evaluate_labels(
        predictions, ground_truth, ignore_list
    )

    map_category_averages = calculate_category_averages(results_meta_data, "MAP", SEARCHAD_CATEGORIES)
    rp_category_averages = calculate_category_averages(results_meta_data, "R-Precision", SEARCHAD_CATEGORIES)

    pk_category_averages = {}
    k_values = [5, 25, 100]
    for k_val in k_values:
        pk_category_averages[f"P@{k_val}"] = calculate_category_averages(
            results_meta_data, f"P@{k_val}", SEARCHAD_CATEGORIES
        )

    _print_results_table(
        results_meta_data,
        overall_map_score,
        overall_mrp_score,
        map_category_averages,
        rp_category_averages,
        overall_pk_results,
        pk_category_averages,
    )

    _save_results_to_csv(
        results_meta_data,
        overall_map_score,
        overall_mrp_score,
        map_category_averages,
        rp_category_averages,
        overall_pk_results,
        pk_category_averages,
        scores_output_dir,
        predictions_file,
    )

    _save_results_to_json(
        results_meta_data,
        overall_map_score,
        overall_mrp_score,
        map_category_averages,
        rp_category_averages,
        overall_pk_results,
        pk_category_averages,
        scores_output_dir,
        predictions_file,
    )
    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SearchAD prediction results against ground truth.")
    parser.add_argument(
        "--predictions-file",
        type=str,
        required=True,
        help="Path to the JSON file containing prediction results.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val"],
        help="Dataset split: 'train' or 'val'.",
    )
    parser.add_argument(
        "--searchad-dir",
        type=str,
        required=True,
        help="Path to the folder where the SearchAD annotation files (ground truth, ignore lists) are located.",
    )
    parser.add_argument(
        "--scores-output-dir",
        type=str,
        required=True,
        help="Path to the folder where the evaluation scores (CSV, JSON) should be saved.",
    )

    args = parser.parse_args()
    import warnings

    warnings.formatwarning = lambda message, *args, **kwargs: f"WARNING: {message}\n"

    try:
        evaluate(
            predictions_file=args.predictions_file,
            split=args.split,
            searchad_dir=args.searchad_dir,
            scores_output_dir=args.scores_output_dir,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Error during evaluation: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)
