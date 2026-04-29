# prune_searchad_datasets.py
import argparse
import os
import warnings
from pathlib import Path

from searchad.utils.io import load_json


def prune_searchad_datasets(searchad_dir: str, datasets_to_prune: list):
    """
    Prunes specified datasets within the SearchAD directory.
    It deletes all files except for specified image files and license files.

    Args:
        searchad_dir (str): The absolute path to the SearchAD directory.
        datasets_to_prune (list): A list of dataset names (e.g., ["acdc", "mapillary_sign"])
                                     to be pruned.

    Raises:
        FileNotFoundError: If searchad_dir, annotation files, or any required image files are not found.
        ValueError: If datasets_to_prune is empty.
    """

    searchad_dir = Path(searchad_dir).resolve()  # Ensure absolute path
    if not searchad_dir.is_dir():
        raise FileNotFoundError(f"SearchAD directory not found: {searchad_dir}")

    if not datasets_to_prune:
        raise ValueError("No datasets specified for pruning.")

    print("--- Starting SearchAD Dataset Pruning ---")
    print(f"SearchAD directory: {searchad_dir}")
    print(f"Datasets to Prune: {', '.join(datasets_to_prune)}")

    # Define annotation/mapping file names relative to searchad_dir
    train_annotations_file = searchad_dir / "searchad_annotations_train.json"
    val_annotations_file = searchad_dir / "searchad_annotations_val.json"
    test_mapping_file = searchad_dir / "searchad_test_mapping_id_to_imagepath.json"

    # Check if annotation files exist
    for f in [train_annotations_file, val_annotations_file, test_mapping_file]:
        if not f.is_file():
            raise FileNotFoundError(f"Required annotation/mapping file not found: {f}")

    # Map dataset names to their respective license file names
    license_files_map = {
        "acdc": "License.pdf",
        "mapillary_sign": "LICENSE.txt",
        "mapillary_vistas": "LICENSE",
        "wd_both02": "license.txt",
        "wd_publicv2p0": "license.txt",
    }

    all_image_paths_to_keep = set()

    # ── 1. Load image paths ──────────────────────────────────────────────────
    print("\nLoading image paths from annotation files...")
    try:
        train_data = load_json(train_annotations_file)
        for rel_path in train_data.keys():
            all_image_paths_to_keep.add(str(searchad_dir / rel_path))
        print(f"Loaded {len(train_data.keys())} image paths from {train_annotations_file.name}.")

        val_data = load_json(val_annotations_file)
        for rel_path in val_data.keys():
            all_image_paths_to_keep.add(str(searchad_dir / rel_path))
        print(f"Loaded {len(val_data.keys())} image paths from {val_annotations_file.name}.")

        test_data = load_json(test_mapping_file)
        for rel_path in test_data.values():
            all_image_paths_to_keep.add(str(searchad_dir / rel_path))
        print(f"Loaded {len(test_data.values())} image paths from {test_mapping_file.name}.")

    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(f"Failed to load annotation files: {e}") from e

    print(f"Total unique image paths identified: {len(all_image_paths_to_keep)}")

    total_datasets_requested = len(datasets_to_prune)
    datasets_skipped_not_found = 0
    datasets_successfully_pruned = 0

    # ── 2. Process each dataset ──────────────────────────────────────────────
    for dataset in datasets_to_prune:
        print(f"\n--- Processing dataset: {dataset} ---")
        dataset_abs_path = searchad_dir / dataset

        if not dataset_abs_path.is_dir():
            warnings.warn(f"Dataset directory not found, skipping: {dataset_abs_path}")
            datasets_skipped_not_found += 1
            continue

        # Identify image paths specific to this dataset
        dataset_images_to_keep = {p for p in all_image_paths_to_keep if p.startswith(str(dataset_abs_path) + os.sep)}

        # Identify the license file path for this dataset
        license_filename = license_files_map.get(dataset)
        full_license_path = None
        if license_filename:
            full_license_path = dataset_abs_path / license_filename
            if not full_license_path.is_file():
                warnings.warn(
                    f"License file '{license_filename}' not found in '{dataset_abs_path}'. It will not be preserved.",
                )
                full_license_path = None  # Ensure it's not considered for preservation if not found
            else:
                print(f"License file to preserve: {full_license_path}")
        else:
            print(f"No license file found for dataset '{dataset}'.")

        # ── 3. Verify image files exist ──────────────────────────────────────
        print(f"Verifying existence of {len(dataset_images_to_keep)} image files for {dataset}...")
        missing_images = []
        for img_path in dataset_images_to_keep:
            if not Path(img_path).is_file():
                missing_images.append(img_path)

        if missing_images:
            raise FileNotFoundError(
                f"Pruning stopped for '{dataset}'. The following image files were not found:\n"
                f"{' '.join(missing_images[:5])}{'...' if len(missing_images) > 5 else ''}"
                f"\nTotal missing: {len(missing_images)}"
            )
        print(f"All {len(dataset_images_to_keep)} image files verified for {dataset}.")

        # ── 4. Delete non-essential files ────────────────────────────────────
        print(f"Starting pruning process for {dataset_abs_path}...")
        files_deleted_count = 0
        dirs_deleted_count = 0  # Keep count internally, but don't print

        # Walk the directory tree bottom-up to delete files and then empty directories
        for root, _dirs, files in os.walk(dataset_abs_path, topdown=False):
            for file_name in files:
                current_file_path = Path(root) / file_name
                if str(current_file_path) not in dataset_images_to_keep and current_file_path != full_license_path:
                    try:
                        os.remove(current_file_path)
                        files_deleted_count += 1
                    except OSError as e:
                        warnings.warn(f"Failed to delete file {current_file_path}: {e}")

            # After deleting files, check if directories are empty and delete them
            # Only remove if it's not the top-level dataset directory itself and it's empty
            if not os.listdir(root) and Path(root) != dataset_abs_path:
                try:
                    os.rmdir(root)
                    dirs_deleted_count += 1
                except OSError as e:
                    warnings.warn(f"Failed to delete empty directory {root}: {e}")
            elif Path(root) == dataset_abs_path and not os.listdir(root):
                # If the top-level dataset directory becomes empty, remove it too
                try:
                    os.rmdir(root)
                    dirs_deleted_count += 1
                except OSError as e:
                    warnings.warn(f"Failed to delete top-level empty directory {root}: {e}")

        print(f"Pruning completed for {dataset}.")
        print(f"Files deleted: {files_deleted_count}")
        datasets_successfully_pruned += 1  # Increment only if pruning for this dataset completes

    print(
        f"\nSummary: {datasets_successfully_pruned} out of {total_datasets_requested} datasets processed successfully."
    )
    if datasets_skipped_not_found > 0:
        print(f"({datasets_skipped_not_found} datasets were skipped because their directories were not found.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prune SearchAD datasets by deleting all files except specified image files and license files."
    )
    parser.add_argument(
        "--searchad-dir",
        type=str,
        required=True,
        help="The absolute path to the SearchAD directory (e.g., /data/SearchAD).",
    )
    parser.add_argument(
        "--datasets-to-prune",
        type=str,
        nargs="+",
        required=True,
        help="List of dataset names to prune (e.g., acdc mapillary_sign). "
        "Available datasets include: acdc, bdd100k_images_100k, cityscapes, ECP, IDD_Segmentation, kitti, "
        "lostandfound, mapillary_sign, mapillary_vistas, nuscenes, wd_both02, wd_publicv2p0.",
    )

    args = parser.parse_args()
    warnings.formatwarning = lambda message, *args, **kwargs: f"WARNING: {message}\n"

    prune_searchad_datasets(args.searchad_dir, args.datasets_to_prune)
    print("Pruning finished.")
