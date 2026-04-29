import argparse
import os
import warnings

from searchad.config.config import SEARCHAD_LABELS, SEARCHAD_TEST_DATASETS
from searchad.utils.checks import (
    check_prediction_dataset_prefixes,
    check_prediction_labels,
    check_prediction_list_lengths,
    check_prediction_paths_in_mapping,
    check_submission_metadata,
)
from searchad.utils.io import load_json, save_json


def create_submission_file(
    searchad_dir: str | os.PathLike,
    predictions_file: str | os.PathLike,
    submission_output_dir: str | os.PathLike,
    team_name: str,
    model_name: str,
    paper_code_affiliation: str,
    search_mode: str,
    searchad_train: str,
    default_queries: str,
    overwrite: bool = False,
):
    """
    Compresses results.json by replacing image paths with IDs using a mapping file,
    and saves the compressed data as submission.json in the specified submission folder.
    It also includes metadata provided by the user in the submission.json.

    This function can be called programmatically or via command-line arguments.

    Args:
        searchad_dir (Union[str, os.PathLike]): Path to the folder containing
                                                  searchad_test_mapping_id_to_imagepath.json.
        predictions_file (Union[str, os.PathLike]): Path to the file containing the JSON results
                                                      (e.g., imagelist with original paths).
        submission_output_dir (Union[str, os.PathLike]): Path to the folder where submission.json will be saved.
        team_name (str): The name of the submitting team.
        model_name (str): The name of the model used for predictions.
        paper_code_affiliation (str): Link to paper/code or affiliation details.
        search_mode (str): The search mode used (Language, Vision, or Multimodal).
        searchad_train (str): "Yes" if SearchAD training dataset was used, "No" otherwise.
        default_queries (str): "Yes" if default queries were used, "No" otherwise.

    Raises:
        FileNotFoundError: If mapping file or results file are not found.
        json.JSONDecodeError: If mapping file or results file are not valid JSON.
        OSError: If there's an issue creating the submission folder.
        Exception: For other unexpected errors during processing.
    """
    # ── 1. Construct full file paths ─────────────────────────────────────────
    mapping_file_path = os.path.join(searchad_dir, "searchad_test_mapping_id_to_imagepath.json")
    submission_file_path = os.path.join(submission_output_dir, "submission.json")

    if os.path.exists(submission_file_path) and not overwrite:
        raise FileExistsError(
            f"Submission file already exists: '{submission_file_path}'. "
            "Either choose a different --submission-output-dir or set overwrite=True (--overwrite)."
        )

    print("--- SearchAD Submission Preparation ---")
    print(f"Mapping file expected at: {mapping_file_path}")
    print(f"Results file to process: {predictions_file}")
    print(f"Submission file will be saved to: {submission_file_path}")

    # ── 2. Load image path to ID mapping ─────────────────────────────────────
    imagepath_to_id: dict[str, int] = {}
    total_images_in_mapping = 0
    loaded_id_to_path_mapping: dict[str, str] = load_json(mapping_file_path)
    for id_str, image_path in loaded_id_to_path_mapping.items():
        imagepath_to_id[image_path] = int(id_str)
    total_images_in_mapping = len(imagepath_to_id)
    print(f"Successfully loaded mapping with {total_images_in_mapping} entries.")

    # ── 3. Load predictions ──────────────────────────────────────────────────
    imagelist_with_paths: dict[str, list[str]] = load_json(predictions_file)
    print(f"Successfully loaded results from '{predictions_file}'.")

    # ── 4. Sanity checks: predictions ────────────────────────────────────────
    print("\n--- Performing Sanity Checks on Predictions File ---")
    predictions_sanity_checks_failed_count = 0

    predictions_sanity_checks_failed_count += check_prediction_list_lengths(
        imagelist_with_paths, total_images_in_mapping
    )
    predictions_sanity_checks_failed_count += check_prediction_labels(imagelist_with_paths, SEARCHAD_LABELS)
    predictions_sanity_checks_failed_count += check_prediction_dataset_prefixes(
        imagelist_with_paths, SEARCHAD_TEST_DATASETS
    )
    predictions_sanity_checks_failed_count += check_prediction_paths_in_mapping(imagelist_with_paths, imagepath_to_id)

    if predictions_sanity_checks_failed_count > 0:
        print(f"--- Predictions Sanity Checks Completed with {predictions_sanity_checks_failed_count} issues found ---")
    else:
        print("--- All Predictions Sanity Checks passed without issues ---")

    # ── 5. Sanity checks: metadata ───────────────────────────────────────────
    print("\n--- Performing Sanity Checks on Metadata ---")
    metadata_sanity_checks_failed_count = check_submission_metadata(
        team_name, model_name, paper_code_affiliation, search_mode, searchad_train, default_queries
    )

    if metadata_sanity_checks_failed_count > 0:
        print(f"--- Metadata Sanity Checks Completed with {metadata_sanity_checks_failed_count} issues found ---")
    else:
        print("--- All Metadata Sanity Checks Passed ---")

    # ── 6. Replace image paths with IDs ──────────────────────────────────────
    imagelist_small: dict[str, list[int]] = {}
    for label, paths_list in imagelist_with_paths.items():
        id_list = []
        for image_path in paths_list:
            if image_path in imagepath_to_id:
                id_list.append(imagepath_to_id[image_path])
            # If an image path was not found in the mapping, it's simply skipped here.
            # The warning for this was already issued in Sanity Check 5.

        imagelist_small[label] = id_list

    print("Image paths successfully replaced with IDs.")

    # ── 7. Create submission data structure ─────────────────────────────────
    submission_data = {
        "metadata": {
            "Team Name": team_name.strip(),
            "Model Name": model_name.strip(),
            "Paper/Code/Affiliation": paper_code_affiliation.strip(),
            "Search Mode": search_mode.capitalize(),  # Ensure consistent capitalization
            "SearchAD Train": searchad_train.capitalize(),  # Ensure "Yes" or "No"
            "Default Queries": default_queries.capitalize(),  # Ensure "Yes" or "No"
        },
        "predictions": imagelist_small,
    }

    # ── 8. Create submission folder ──────────────────────────────────────────
    try:
        os.makedirs(submission_output_dir, exist_ok=True)
        print(f"Ensured submission folder '{submission_output_dir}' exists.")
    except OSError as e:
        raise OSError(f"Error: Could not create submission folder '{submission_output_dir}': {e}") from e

    # ── 9. Save submission.json ───────────────────────────────────────────────
    save_json(submission_data, submission_file_path, indent=None)
    print(f"Successfully saved compressed results and metadata to '{submission_file_path}'.")

    total_failed_checks = predictions_sanity_checks_failed_count + metadata_sanity_checks_failed_count

    if total_failed_checks > 0:
        print(
            f"\nProcess complete, but with {total_failed_checks} issues detected during sanity checks. "
            f"Please review the warnings above carefully before submitting your 'submission.json' file to "
            f"https://huggingface.co/spaces/SearchADBenchmark/SearchADLargeScaleRareImageRetrievalDatasetforAutonomousDriving"
        )
    else:
        print(
            "\nProcess complete. Submission file should be available at your submission folder and can be uploaded to "
            "https://huggingface.co/spaces/SearchADBenchmark/SearchADLargeScaleRareImageRetrievalDatasetforAutonomousDriving"
        )
    print(
        "\nNOTE: On the benchmark page, click the blue 'Login with Hugging Face' button "
        "in the bottom left before submitting. If the button does not respond, try a different browser "
        "(Safari has been observed to silently fail)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress retrieval results by replacing image paths with IDs for submission, including metadata."
    )
    parser.add_argument(
        "--searchad-dir",
        type=str,
        required=True,
        help="Path to the folder containing 'searchad_test_mapping_id_to_imagepath.json'.",
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        required=True,
        help="Path to the JSON file containing the retrieval results (imagelist with original paths).",
    )
    parser.add_argument(
        "--submission-output-dir",
        type=str,
        required=True,
        help="Path to the folder where 'submission.json' will be saved.",
    )
    parser.add_argument(
        "--team-name",
        type=str,
        required=True,
        help="Your team's name. Must be a non-empty string.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="The name of your model. Must be a non-empty string.",
    )
    parser.add_argument(
        "--paper-code-affiliation",
        type=str,
        required=True,
        help="Link to your paper/code or your affiliation. Must be a non-empty string.",
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        required=True,
        choices=["Language", "Vision", "Multimodal"],
        help="The search mode used by your model. Must be one of 'Language', 'Vision', or 'Multimodal'.",
    )
    parser.add_argument(
        "--searchad-train",
        type=str,
        required=True,
        choices=["Yes", "No"],
        help="Did you use the SearchAD training dataset? Must be 'Yes' or 'No'.",
    )
    parser.add_argument(
        "--default-queries",
        type=str,
        required=True,
        choices=["Yes", "No"],
        help="Did you use the default queries provided by SearchAD? Must be 'Yes' or 'No'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite an existing submission.json. Default: Script exits with an error if file exists.",
    )

    args = parser.parse_args()
    warnings.formatwarning = lambda message, *args, **kwargs: f"WARNING: {message}\n"

    create_submission_file(
        searchad_dir=args.searchad_dir,
        predictions_file=args.predictions_file,
        submission_output_dir=args.submission_output_dir,
        team_name=args.team_name,
        model_name=args.model_name,
        paper_code_affiliation=args.paper_code_affiliation,
        search_mode=args.search_mode,
        searchad_train=args.searchad_train,
        default_queries=args.default_queries,
        overwrite=args.overwrite,
    )
