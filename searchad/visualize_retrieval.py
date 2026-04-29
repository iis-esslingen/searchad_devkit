import argparse
import os
import warnings
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt

# Import configuration constants from your project's config file
from searchad.config.config import (
    BBOX_LINE_WIDTH,
    BORDER_WIDTH,
    COLLAGE_H_PAD,
    COLLAGE_RESIZE_DIM,
    COLLAGE_SUBTITLE_FONTSIZE,
    COLLAGE_TITLE_FONTSIZE,
    COLLAGE_W_PAD,
    DEFAULT_COLOR_RGB,
    FONT_SCALE,
    FONT_THICKNESS,
    LABEL_COLORS,
    MIN_BOX_AREA,
    SEARCHAD_DATASETS,
    TEXT_PADDING,
)
from searchad.utils.io import load_json, subdataset_for_path
from searchad.utils.visualization import draw_image_with_annotations, save_placeholder


def _load_data(
    predictions_file: str, searchad_dir: str, split: str, min_box_area: int
) -> tuple[dict, dict, dict, dict]:
    """
    Loads prediction, ground truth, ignore list, and object-level annotation data from JSON files.
    Assumes annotations are in searchad_annotations_{split}.json and
    ignore list in searchad_{split}_image_level_ignore_images_min_box_{MIN_BOX_AREA}.json.
    """
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    # Consolidated annotations file (as per your last correction)
    all_annotations_filepath = os.path.join(searchad_dir, f"searchad_annotations_{split}.json")
    # Corrected ignore list filename (re-introducing min_box_area)
    ignore_file_path = os.path.join(
        searchad_dir, f"searchad_{split}_image_level_ignore_images_min_box_{min_box_area}.json"
    )

    if not os.path.exists(all_annotations_filepath):
        raise FileNotFoundError(f"All annotations file not found: {all_annotations_filepath}")

    ignore_list = load_json(ignore_file_path) if os.path.exists(ignore_file_path) else {}
    if not ignore_list:
        print(f"Warning: Ignore list file not found: {ignore_file_path}. Proceeding without ignore list.")

    predictions = load_json(predictions_file)
    all_annotations_raw = load_json(all_annotations_filepath)

    # object_annotations is directly the loaded raw annotations
    object_annotations = all_annotations_raw

    # Derive image-level ground_truth from all_annotations_raw
    ground_truth = defaultdict(list)
    for relative_image_path, annotations_list in all_annotations_raw.items():
        for annotation in annotations_list:
            label = annotation.get("label")
            if label:  # Ensure label exists
                ground_truth[label].append(relative_image_path)

    # Convert defaultdict to regular dict for consistency
    ground_truth = dict(ground_truth.items())

    return predictions, ground_truth, ignore_list, object_annotations


def _create_collage(
    image_paths: list[str],
    query_label: str,
    output_dir: str,  # This is the query_output_dir
    topk: int,
) -> None:
    """
    Creates a collage of the top-k retrieved images using matplotlib.
    Images are expected to be pre-resized if resize_for_collage was True in visualize_retrieval.
    """
    if not image_paths:
        print(f"No images to create collage for query '{query_label}'.")
        return

    # Determine grid size for collage
    max_cols = 5
    cols = min(topk, max_cols)
    rows = (topk + cols - 1) // cols  # Calculate rows needed

    # ── Compute figure dimensions from first principles ──────────────────────
    img_w_in = COLLAGE_RESIZE_DIM[0] / 100  # one image cell width  (px → inches)
    img_h_in = COLLAGE_RESIZE_DIM[1] / 100  # one image cell height (px → inches)
    subtitle_in = COLLAGE_SUBTITLE_FONTSIZE / 72 * 1.5  # pt → inches, per-row subtitle
    title_in = COLLAGE_TITLE_FONTSIZE / 72 * 1.5  # pt → inches, suptitle

    # fig dimensions: image cells  +  inter-cell gaps  +  text rows
    # Changing COLLAGE_W_PAD affects only fig_width and wspace — nothing else.
    # Changing COLLAGE_H_PAD affects only fig_height and hspace — nothing else.
    fig_width = cols * img_w_in + (cols - 1) * COLLAGE_W_PAD
    fig_height = rows * img_h_in + rows * subtitle_in + (rows - 1) * COLLAGE_H_PAD + title_in

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # subplots_adjust uses fractions of figure size; derive them from the inch values above
    wspace = COLLAGE_W_PAD / img_w_in  # col gap as fraction of axes width
    hspace = (COLLAGE_H_PAD + subtitle_in) / img_h_in  # row gap + subtitle as fraction of axes height
    top = 1.0 - title_in / fig_height  # fraction reserved at top for suptitle
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=top, wspace=wspace, hspace=hspace)

    # Flatten axes array for easy iteration, handles both 1D and 2D cases
    if rows == 1 and cols == 1:
        axes = [axes]  # Make it iterable if only one subplot
    elif rows == 1 or cols == 1:
        axes = axes  # Already 1D if one row or one col
    else:
        axes = axes.flatten()

    for i, img_path in enumerate(image_paths):
        if i >= topk:  # Ensure we don't plot more than topk if image_paths has more
            break

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found or could not be read: {img_path}")

            # Convert BGR to RGB for matplotlib
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # No need to resize here, as _draw_image_with_annotations already handled it
            # if resize_images was True. The images loaded here are already the target size.

            axes[i].imshow(img)
            axes[i].set_title(f"Top{i + 1}", fontsize=COLLAGE_SUBTITLE_FONTSIZE)
            axes[i].axis("off")  # Hide axes ticks
        except FileNotFoundError:
            print(f"Warning: Image for collage not found: {img_path}. Skipping.")
            axes[i].axis("off")  # Still hide axis for empty plot
        except Exception as e:
            warnings.warn(f"Error loading image {img_path} for collage: {e}. Skipping.")
            axes[i].axis("off")

    # Hide any unused subplots
    for j in range(len(image_paths), len(axes)):
        axes[j].axis("off")

    # Updated title format
    # plt.suptitle(f"Top-{topk} Retrieval for Query: {query_label}", fontsize=COLLAGE_TITLE_FONTSIZE)

    collage_output_filename = f"collage_{query_label.lower().replace('/', '_')}_top{topk}.jpg"
    collage_output_path = os.path.join(output_dir, collage_output_filename)
    plt.savefig(collage_output_path, bbox_inches="tight", dpi=150)  # Save with high DPI
    plt.close(fig)  # Close the figure to free memory
    print(f"Collage saved to: {collage_output_path}")


def visualize_retrieval(
    predictions_file: str,
    searchad_dir: str,
    visualization_output_dir: str,
    split: str,
    topk: int,
    searchad_label: str | None = None,
    resize_for_collage: bool = False,
    shorten_labels: bool = True,
) -> None:
    """
    Visualizes the top-k retrieved images for each query, with correctness borders and bounding boxes.
    If searchad_label is provided, only visualizes for that specific label.
    """
    label_str = searchad_label if searchad_label is not None else "all"
    print(f"Starting retrieval visualization for split '{split}', top-k={topk}, label={label_str}...")
    print(f"Loading data from: {searchad_dir}")

    predictions, ground_truth, ignore_list, object_annotations = _load_data(
        predictions_file,
        searchad_dir,
        split,
        MIN_BOX_AREA,  # Pass min_box_area to _load_data
    )

    os.makedirs(visualization_output_dir, exist_ok=True)

    # Pre-process ground_truth and ignore_list into sets for faster lookup
    processed_ground_truth = {label: set(paths) for label, paths in ground_truth.items()}
    processed_ignore_list = {label: set(paths) for label, paths in ignore_list.items()}

    # Determine which labels to visualize
    labels_to_visualize = []
    if searchad_label:
        if searchad_label in predictions:
            labels_to_visualize.append(searchad_label)
        else:
            print(
                f"Warning: Target label '{searchad_label}' not found in predictions. "
                f"No visualizations will be generated for this label."
            )
            return
    else:
        labels_to_visualize = sorted(predictions.keys())

    for query_label in labels_to_visualize:
        predicted_image_paths = predictions[query_label]

        # Create query-specific output directory
        query_output_dir = os.path.join(visualization_output_dir, query_label.replace("/", "_"))
        os.makedirs(query_output_dir, exist_ok=True)

        # Create new subfolder for retrieved images within the query directory
        retrieved_images_subfolder = os.path.join(query_output_dir, "retrieved_images")
        os.makedirs(retrieved_images_subfolder, exist_ok=True)

        gt_image_paths_set = processed_ground_truth.get(query_label, set())
        ignore_image_paths_set = processed_ignore_list.get(query_label, set())

        # Select top-k images
        topk_images = predicted_image_paths[:topk]

        # List to store paths of individually saved images for collage creation
        saved_image_paths_for_collage = []

        for i, image_path in enumerate(topk_images):
            rank = i + 1
            # Construct full path to the image file
            # Assuming image_path in JSON is relative to searchad_dir or is an absolute path
            image_full_path = os.path.join(searchad_dir, image_path)
            if not os.path.exists(image_full_path):
                # If os.path.join didn't work, try assuming image_path is already absolute
                image_full_path = image_path
                if not os.path.exists(image_full_path):
                    print(f"  [missing] {image_path}")
                    placeholder_path = os.path.join(retrieved_images_subfolder, f"top_{rank}_not_found.jpg")
                    save_placeholder(placeholder_path, COLLAGE_RESIZE_DIM)
                    saved_image_paths_for_collage.append(placeholder_path)
                    continue

            is_correct_retrieval = image_path in gt_image_paths_set and image_path not in ignore_image_paths_set

            # Get object annotations for the current image
            # The keys in object_annotations should match the image_path format in predictions
            annotations_for_current_image = object_annotations.get(image_path, [])

            # Determine subdataset name for output filename
            subdataset_name = subdataset_for_path(image_path, SEARCHAD_DATASETS) or "unknown_subdataset"

            # New naming convention: top_X_subdataset_original_filename.jpg
            original_filename = os.path.basename(image_path)
            output_image_filename = f"top_{rank}_{subdataset_name}_{original_filename}"
            output_image_path = os.path.join(retrieved_images_subfolder, output_image_filename)

            draw_image_with_annotations(
                image_full_path=image_full_path,
                object_annotations_for_image=annotations_for_current_image,
                output_path=output_image_path,
                label_colors=LABEL_COLORS,
                default_color_rgb=DEFAULT_COLOR_RGB,
                bbox_line_width=BBOX_LINE_WIDTH,
                font_scale=FONT_SCALE,
                font_thickness=FONT_THICKNESS,
                text_padding=TEXT_PADDING,
                is_correct_retrieval=is_correct_retrieval,
                border_width=BORDER_WIDTH,
                resize_for_drawing=resize_for_collage,
                target_resize_dim=COLLAGE_RESIZE_DIM,
                shorten_labels=shorten_labels,
            )
            saved_image_paths_for_collage.append(output_image_path)  # Collect path for collage

        # After processing all top-k images for the current query, create the collage
        _create_collage(saved_image_paths_for_collage, query_label, query_output_dir, topk)

    print("Retrieval visualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize top-k retrieved images with correctness borders and bounding boxes."
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        required=True,
        help="Path to the JSON file containing prediction results (query_label -> [image_paths]).",
    )
    parser.add_argument(
        "--searchad-dir",
        type=str,
        required=True,
        help="Path to the folder containing SearchAD annotation files and images.",
    )
    parser.add_argument(
        "--visualization-output-dir",
        type=str,
        required=True,
        help="Path to the folder where visualized images will be saved.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val"],
        help="Dataset split: 'train' or 'val'.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of top retrieved images to visualize for each query.",
    )
    parser.add_argument(
        "--searchad-label",
        type=str,
        default=None,
        help="Optional: Specify a single label to visualize. If not provided, all labels will be visualized.",
    )
    parser.add_argument(
        "--resize-for-collage",
        action="store_true",  # This makes it a boolean flag
        help="If set, images in the collage will be resized to a consistent 16:9 aspect ratio.",
    )

    parser.add_argument(
        "--shorten-labels",
        action="store_true",
        help="If set, display shortened label names instead of full ones.",
    )

    args = parser.parse_args()
    warnings.formatwarning = lambda message, *args, **kwargs: f"WARNING: {message}\n"

    try:
        visualize_retrieval(
            predictions_file=args.predictions_file,
            searchad_dir=args.searchad_dir,
            visualization_output_dir=args.visualization_output_dir,
            split=args.split,
            topk=args.topk,
            searchad_label=args.searchad_label,
            resize_for_collage=args.resize_for_collage,
            shorten_labels=args.shorten_labels,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)
