import argparse
import os
import random

from searchad.config.config import (
    BBOX_LINE_WIDTH,
    DEFAULT_COLOR_RGB,
    FONT_SCALE,
    FONT_THICKNESS,
    LABEL_COLORS,
    SEARCHAD_DATASETS,
    SEARCHAD_LABELS,
    TEXT_PADDING,
)
from searchad.utils.io import load_json, subdataset_for_path
from searchad.utils.visualization import draw_image_with_annotations


def _load_and_select_images_with_label(
    searchad_dir: str | os.PathLike,
    split: str,
    searchad_label: str,
    num_images: int,
) -> dict[str, tuple[str, list[dict]]]:  # Changed return type
    """
    Loads annotations, identifies images containing the searchad_label,
    randomly selects a specified number of these images, and returns
    all relevant annotations for the selected images, along with their relative paths.

    Args:
        searchad_dir (Union[str, os.PathLike]): The path to the SearchAD directory.
        split (str): The dataset split ('train' or 'val').
        searchad_label (str): The specific label that selected images must contain.
        num_images (int): The number of images to randomly select and visualize.

    Returns:
        Dict[str, Tuple[str, List[Dict]]]: A dictionary where keys are full image paths of selected images
                                           and values are a tuple: (relative_image_path, list_of_annotations),
                                           where list_of_annotations contains 'bbox' and 'label'
                                           for all relevant labels found in that image.
    """
    annotations_filepath = os.path.join(searchad_dir, f"searchad_annotations_{split}.json")

    print(f"Loading annotations for split '{split}' from {annotations_filepath}...")
    all_annotations_raw = load_json(annotations_filepath)
    images_containing_searchad_label = []
    for relative_image_path, annotations_list in all_annotations_raw.items():
        for annotation in annotations_list:
            if annotation.get("label") == searchad_label:
                images_containing_searchad_label.append(relative_image_path)
                break  # Found searchad_label in this image, move to next image

    if not images_containing_searchad_label:
        print(f"No images found containing the label '{searchad_label}'.")
        return {}

    print(f"Total images found containing label '{searchad_label}': {len(images_containing_searchad_label)}.")

    # Randomly select num_images, handling cases where num_images > available images.
    num_to_select = min(num_images, len(images_containing_searchad_label))
    selected_relative_image_paths = random.sample(images_containing_searchad_label, num_to_select)
    print(f"Randomly selected {num_to_select} images for visualization.")

    final_selected_annotations = {}  # Changed to dict directly
    for relative_image_path in selected_relative_image_paths:
        image_full_path = os.path.join(searchad_dir, relative_image_path)
        current_image_annotations = []
        for annotation in all_annotations_raw[relative_image_path]:
            label = annotation.get("label")
            # Only include labels that are part of SEARCHAD_LABELS and have a color defined
            if label in SEARCHAD_LABELS and label in LABEL_COLORS:
                bbox_data = annotation.get("bbox")
                if bbox_data and isinstance(bbox_data, list) and len(bbox_data) == 4:
                    current_image_annotations.append({"bbox": bbox_data, "label": label})
                else:
                    print(
                        f"Warning: Skipping malformed bbox for label '{label}' "
                        f"in image '{relative_image_path}': {bbox_data}"
                    )
        if current_image_annotations:  # Only add if there are valid annotations for this image
            final_selected_annotations[image_full_path] = (
                relative_image_path,
                current_image_annotations,
            )
    return final_selected_annotations


def visualize_bbox_labels(
    searchad_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    searchad_label: str | None,
    split: str,
    num_images: int,
    shorten_labels: bool = True,
    only_target_label: bool = False,
    show_labels: bool = True,
):
    labels_to_visualize = [searchad_label] if searchad_label else SEARCHAD_LABELS

    for label in labels_to_visualize:
        _visualize_bbox_labels_for_label(
            searchad_dir=searchad_dir,
            output_dir=output_dir,
            searchad_label=label,
            split=split,
            num_images=num_images,
            shorten_labels=shorten_labels,
            only_target_label=only_target_label,
            show_labels=show_labels,
        )


def _visualize_bbox_labels_for_label(
    searchad_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    searchad_label: str,
    split: str,
    num_images: int,
    shorten_labels: bool = True,
    only_target_label: bool = False,
    show_labels: bool = True,
):
    print(f"--- SearchAD Bounding Box Visualization for Label: '{searchad_label}' ---")
    print(f"SearchAD directory: {searchad_dir}")
    print(f"Annotations split:  {split}")
    print(f"Parent Output directory:   {output_dir}")
    print(f"Target label for selection: '{searchad_label}'")
    print(f"Number of images to visualize: {num_images}")

    # Create the label-specific output directory
    dir_suffix = ("_only" if only_target_label else "") + ("_nolabels" if not show_labels else "")
    label_specific_output_dir = os.path.join(output_dir, f"{searchad_label}{dir_suffix}")
    os.makedirs(label_specific_output_dir, exist_ok=True)
    print(f"Saving visualizations to: {label_specific_output_dir}")

    selected_annotations = _load_and_select_images_with_label(searchad_dir, split, searchad_label, num_images)

    if not selected_annotations:
        print("No images selected for visualization. Exiting.")
        return

    images_processed = 0
    for image_full_path, (
        relative_image_path,
        annotations_for_image,
    ) in selected_annotations.items():
        if only_target_label:
            annotations_for_image = [a for a in annotations_for_image if a.get("label") == searchad_label]
        subdataset_name = subdataset_for_path(relative_image_path, SEARCHAD_DATASETS)
        if subdataset_name is None:
            print(
                f"  Warning: Could not determine subdataset for relative path '{relative_image_path}'. "
                f"Using 'unknown_subdataset'."
            )
            subdataset_name = "unknown_subdataset"
        output_path = os.path.join(
            label_specific_output_dir,
            f"{subdataset_name}_{os.path.basename(image_full_path)}",
        )
        draw_image_with_annotations(
            image_full_path=image_full_path,
            object_annotations_for_image=annotations_for_image,
            output_path=output_path,
            label_colors=LABEL_COLORS,
            default_color_rgb=DEFAULT_COLOR_RGB,
            bbox_line_width=BBOX_LINE_WIDTH,
            font_scale=FONT_SCALE,
            font_thickness=FONT_THICKNESS,
            text_padding=TEXT_PADDING,
            shorten_labels=shorten_labels,
            show_labels=show_labels,
        )
        images_processed += 1

    print(f"\nVisualization complete. {images_processed} image(s) processed.")
    print(f"Annotated images saved to: {label_specific_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize SearchAD bounding boxes for a specific label, "
        "randomly selecting N images that contain it, and plotting all relevant bboxes."
    )
    parser.add_argument(
        "--searchad-dir",
        type=str,
        required=True,
        help="Path to the SearchAD directory. Annotations are read from "
        "<searchad-dir>/searchad_annotations_{split}.json and image paths are resolved "
        "relative to this directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the parent directory where label-specific output folders will be created.",
    )
    parser.add_argument(
        "--searchad-label",
        type=str,
        default=None,
        help="The specific label (e.g., 'Animal-Real-Cat') that images must contain for selection. "
        "If not provided, all SearchAD labels will be visualized.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val"],
        help="Dataset split: 'train' or 'val'.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to randomly select and visualize that contain the specified label. Default: 10.",
    )

    parser.add_argument(
        "--shorten-labels",
        action="store_true",
        help="If set, display shortened label names instead of full ones.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="If set, draw bounding boxes without any label text.",
    )
    parser.add_argument(
        "--only-target-label",
        action="store_true",
        help="If set, only draw bounding boxes for the target label; all other labels in the image are suppressed.",
    )

    args = parser.parse_args()

    try:
        visualize_bbox_labels(
            searchad_dir=args.searchad_dir,
            output_dir=args.output_dir,
            searchad_label=args.searchad_label,
            split=args.split,
            num_images=args.num_images,
            shorten_labels=args.shorten_labels,
            only_target_label=args.only_target_label,
            show_labels=not args.hide_labels,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        exit(1)
