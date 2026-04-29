import argparse
import os
import warnings
from collections import defaultdict

from searchad.config.config import (
    IGNORE_SIZE_LABELS,
    MIN_BOX_AREA,
    SEARCHAD_DATASETS,
    SEARCHAD_LABELS,
)
from searchad.utils.io import load_json, save_json


def load_annotations_and_filter_size(
    dataset_folder: str | os.PathLike, split: str
) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    """
    Loads annotations from a JSON file and processes them to identify images
    with objects meeting the minimum box area criteria, and images to ignore.

    Args:
        dataset_folder (Union[str, os.PathLike]): The path to the dataset folder containing the annotations file.
        split (str): The dataset split ('train' or 'val').

    Returns:
        Tuple[Dict[str, List[dict]], Dict[str, List[str]]]:
            A tuple containing:
            - annotations_wo_bbox (Dict[str, List[dict]]): A dictionary where keys are labels
              and values are lists of dictionaries, each containing 'image' path and 'label'.
            - ignore_images (Dict[str, List[str]]): A dictionary where keys are labels
              and values are lists of image paths to be ignored for that label.
    """
    annotations_wo_bbox = defaultdict(list)
    annotations_file = f"searchad_annotations_{split}.json"
    ignore_images = defaultdict(list)

    annotations_filepath = os.path.join(dataset_folder, annotations_file)

    if not os.path.exists(annotations_filepath):
        raise FileNotFoundError(f"Annotations file not found: {annotations_filepath}")

    annotation_dict = load_json(annotations_filepath)

    for image_path, annotations in annotation_dict.items():
        classes_with_at_least_one_valid_object = []
        count_valid_objects = defaultdict(int)
        for annotation in annotations:
            label = annotation["label"]
            if label not in ignore_images.keys():
                ignore_images[label] = []
            if label not in count_valid_objects.keys():
                count_valid_objects[label] = 0

        for annotation in annotations:
            label = annotation["label"]
            box_list = annotation["bbox"]
            bbox_area = (box_list[2] - box_list[0]) * (box_list[3] - box_list[1])

            if bbox_area >= MIN_BOX_AREA or label in IGNORE_SIZE_LABELS:
                count_valid_objects[label] += 1

                if label not in classes_with_at_least_one_valid_object:
                    annotation_wo_bbox = {"image": image_path, "label": label}
                    annotations_wo_bbox[label].append(annotation_wo_bbox)
                    classes_with_at_least_one_valid_object.append(label)

        for label, count in count_valid_objects.items():
            if count == 0:
                if image_path not in ignore_images[label]:
                    ignore_images[label].append(image_path)
                print(f"Ignoring {image_path} due to bbox smaller than {MIN_BOX_AREA} pixels for label {label}")
    return annotations_wo_bbox, ignore_images


def filter_benchmark_classes(
    annotations_wo_bbox: dict[str, list[dict]],
    ignore_images_raw: dict[str, list[str]],
    benchmark_labels: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Filters the loaded annotations and ignore lists to include only labels
    that are part of the specified SearchAD benchmark.

    Args:
        annotations_wo_bbox (Dict[str, List[dict]]): Annotations loaded from the dataset.
        ignore_images_raw (Dict[str, List[str]]): Raw ignore list from the dataset.
        benchmark_labels (List[str]): A list of labels considered part of the SearchAD benchmark.

    Returns:
        Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
            A tuple containing:
            - searchad_image_level_annotations (Dict[str, List[str]]): Filtered image paths per benchmark label.
            - searchad_ignore_images (Dict[str, List[str]]): Filtered ignore image paths per benchmark label.
    """
    searchad_image_level_annotations = defaultdict(list)
    searchad_ignore_images = defaultdict(list)

    # Process annotations_wo_bbox
    for label_from_annotations, annotations_list in annotations_wo_bbox.items():
        if label_from_annotations not in benchmark_labels:
            print(f"Label '{label_from_annotations}' not part of SearchAD Benchmark, skipping...")
            continue
        for annotation in annotations_list:
            searchad_image_level_annotations[label_from_annotations].append(annotation["image"])

    # Process ignore_images_raw
    for label_from_ignore, images_to_ignore_list in ignore_images_raw.items():
        if label_from_ignore not in benchmark_labels:
            # print(f"Label '{label_from_ignore}' in ignore list but not in SearchAD Benchmark, skipping...")
            continue
        # Add to searchad_ignore_images, avoiding duplicates if already added from annotations_wo_bbox processing
        for img_path in images_to_ignore_list:
            if img_path not in searchad_ignore_images[label_from_ignore]:
                searchad_ignore_images[label_from_ignore].append(img_path)

    return dict(searchad_image_level_annotations), dict(searchad_ignore_images)


def prepare_image_level_annotations(searchad_dir: str | os.PathLike, split: str):
    """
    Prepares image-level annotations and ignore lists for the SearchAD benchmark.
    This function can be called programmatically.

    Args:
        searchad_dir (Union[str, os.PathLike]): The path to the SearchAD annotations directory.
        split (str): The dataset split ('train' or 'val').

    Raises:
        ValueError: If an invalid split is provided.
        FileNotFoundError: If the annotations file does not exist.
    """
    if split not in ["train", "val"]:
        raise ValueError("Invalid split. Please use 'train' or 'val'.")

    print(f"Loading annotations for split '{split}' from {searchad_dir}...")
    annotations_wo_bbox, ignore_images = load_annotations_and_filter_size(searchad_dir, split)
    print("Annotations loaded. Filtering for labels included in SearchAD Benchmark...")
    searchad_image_level_annotations, searchad_ignore_images = filter_benchmark_classes(
        annotations_wo_bbox, ignore_images, SEARCHAD_LABELS
    )
    print(
        "Filtering complete. Saving image-level annotations and ignore lists to JSON files in the SearchAD directory..."
    )

    output_annotations_path = os.path.join(
        searchad_dir, f"searchad_{split}_image_level_annotations_min_box_{MIN_BOX_AREA}.json"
    )
    output_ignore_path = os.path.join(
        searchad_dir, f"searchad_{split}_image_level_ignore_images_min_box_{MIN_BOX_AREA}.json"
    )

    save_json(searchad_image_level_annotations, output_annotations_path)
    print(f"Image-level annotations saved to: {output_annotations_path}")

    save_json(searchad_ignore_images, output_ignore_path)
    print(f"Ignore list saved to: {output_ignore_path}")

    # ── Image presence summary ───────────────────────────────────────────────
    all_referenced = {img for paths in searchad_image_level_annotations.values() for img in paths}
    missing_datasets = []
    print(f"\nImage presence summary ({len(all_referenced)} unique images referenced):")
    for ds in SEARCHAD_DATASETS:
        ds_paths = [p for p in all_referenced if p.startswith(ds + "/")]
        if not ds_paths:
            continue
        found = sum(1 for p in ds_paths if os.path.isfile(os.path.join(searchad_dir, p)))
        status = "OK" if found == len(ds_paths) else "MISSING"
        print(f"  [{status:7s}] {ds}/  {found}/{len(ds_paths)} images on disk")
        if found < len(ds_paths):
            missing_datasets.append(ds)

    if missing_datasets:
        warnings.warn(
            f"{len(missing_datasets)} dataset(s) have images missing from disk: "
            f"{missing_datasets}. "
            f"The generated annotation files are complete, but evaluation and "
            f"visualization will fail for those datasets until their images are present."
        )

    print(f"Image-level annotations and ignore lists saved successfully for split '{split}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare image-level annotations and ignore lists for the SearchAD benchmark."
    )
    parser.add_argument(
        "--searchad-dir",
        type=str,
        required=True,
        help="Path to the SearchAD annotations directory (e.g., /path/to/searchad).",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val"],
        help="Dataset split: 'train' or 'val'.",
    )

    args = parser.parse_args()
    warnings.formatwarning = lambda message, *args, **kwargs: f"WARNING: {message}\n"

    try:
        prepare_image_level_annotations(searchad_dir=args.searchad_dir, split=args.split)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        exit(1)
