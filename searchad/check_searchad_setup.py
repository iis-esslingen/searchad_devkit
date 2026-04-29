import argparse
import os

from searchad.config.config import MIN_BOX_AREA, SEARCHAD_DATASETS, SEARCHAD_LABELS
from searchad.utils.checks import check_annotation_label_coverage, check_query_file_coverage
from searchad.utils.io import load_json


def check_searchad_setup(searchad_dir: str | os.PathLike) -> bool:
    """
    Checks the completeness of a SearchAD dataset directory.

    Verifies the presence of:
      - Required JSON annotation and mapping files
      - Image-level ignore list files
      - default_queries/ directory with at least one .json query file per label
      - All expected subdataset image directories

    Args:
        searchad_dir: Path to the SearchAD directory.

    Returns:
        True if all checks pass, False if any issues are found.
    """
    searchad_dir = os.path.abspath(searchad_dir)
    print("=" * 60)
    print("SearchAD Setup Check")
    print(f"Directory: {searchad_dir}")
    print("=" * 60)

    ok = True
    failures = 0

    # ── 1. SearchAD directory ──────────────────────────────────────────────────
    print("\n[1] SearchAD directory")
    if not os.path.isdir(searchad_dir):
        print(f"  FAIL  Directory not found: {searchad_dir}")
        return False
    print(f"  OK    {searchad_dir}")

    # ── 2. Required JSON files ───────────────────────────────────────────────
    print("\n[2] Required JSON files")
    required_files = [
        "searchad_annotations_train.json",
        "searchad_annotations_val.json",
        "searchad_test_mapping_id_to_imagepath.json",
    ]
    for filename in required_files:
        path = os.path.join(searchad_dir, filename)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  OK    {filename}  ({size_mb:.1f} MB)")
        else:
            print(f"  FAIL  {filename}  (not found)")
            ok = False
            failures += 1

    # ── 2b. Optional generated files ────────────────────────────────────────
    print("\n[2b] Optional generated files (run prepare_image_level_annotations.py to create)")
    optional_files = [
        f"searchad_val_image_level_annotations_min_box_{MIN_BOX_AREA}.json",
        f"searchad_val_image_level_ignore_images_min_box_{MIN_BOX_AREA}.json",
    ]
    for filename in optional_files:
        path = os.path.join(searchad_dir, filename)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  OK    {filename}  ({size_mb:.1f} MB)")
        else:
            print(f"  INFO  {filename}  (not yet generated)")

    # ── 3. Annotation file sanity (spot-check label coverage) ───────────────
    print("\n[3] Annotation file sanity")
    for split in ("train", "val"):
        ann_path = os.path.join(searchad_dir, f"searchad_annotations_{split}.json")
        if not os.path.isfile(ann_path):
            print(f"  SKIP  Cannot check {split} annotations (file missing)")
            continue
        data = load_json(ann_path)
        missing = check_annotation_label_coverage(data, SEARCHAD_LABELS)
        if missing:
            print(f"  WARN  {split}: {len(missing)} label(s) have no annotations: {missing}")
            ok = False
            failures += 1
        else:
            print(f"  OK    {split}: all {len(SEARCHAD_LABELS)} labels present  ({len(data)} images)")

    # ── 4. Default queries directory ─────────────────────────────────────────
    print("\n[4] Default queries directory")
    queries_dir = os.path.join(searchad_dir, "default_queries")
    if not os.path.isdir(queries_dir):
        print("  FAIL  default_queries/ directory not found")
        ok = False
        failures += 1
    else:
        missing_queries, total_query_files = check_query_file_coverage(queries_dir, SEARCHAD_LABELS)
        if not total_query_files:
            print("  FAIL  default_queries/ exists but contains no .json files")
            ok = False
            failures += 1
        elif missing_queries:
            print(f"  WARN  {len(missing_queries)} label(s) have no query file: {missing_queries}")
            ok = False
            failures += 1
        else:
            print(f"  OK    {total_query_files} query files found (all labels covered)")
    # ── 5. Subdataset image directories and image paths ──────────────────────
    print("\n[5] Subdataset image directories and image paths")

    # Collect all unique relative image paths from annotation files and test mapping
    all_rel_paths: set[str] = set()
    split_counts: dict[str, int] = {}
    for split in ("train", "val"):
        ann_path = os.path.join(searchad_dir, f"searchad_annotations_{split}.json")
        if os.path.isfile(ann_path):
            data = load_json(ann_path)
            split_paths = set(data.keys())
            split_counts[split] = len(split_paths)
            all_rel_paths.update(split_paths)

    test_mapping_path = os.path.join(searchad_dir, "searchad_test_mapping_id_to_imagepath.json")
    if os.path.isfile(test_mapping_path):
        test_mapping = load_json(test_mapping_path)
        test_paths = set(test_mapping.values())
        split_counts["test"] = len(test_paths)
        all_rel_paths.update(test_paths)

    split_summary = ", ".join(f"{s}: {n}" for s, n in split_counts.items())
    print(f"  INFO  Total unique image paths referenced: {len(all_rel_paths)}  ({split_summary})")

    missing_datasets = []
    for ds in SEARCHAD_DATASETS:
        ds_path = os.path.join(searchad_dir, ds)
        if not os.path.isdir(ds_path):
            print(f"  FAIL  {ds}/  (directory not found)")
            missing_datasets.append(ds)
            ok = False
            failures += 1
            continue

        ds_paths = [p for p in all_rel_paths if p.startswith(ds + os.sep) or p.startswith(ds + "/")]
        if not ds_paths:
            print(f"  OK    {ds}/  (no images referenced in annotations)")
            continue

        found = sum(1 for p in ds_paths if os.path.isfile(os.path.join(searchad_dir, p)))
        missing_count = len(ds_paths) - found
        if missing_count == 0:
            print(f"  OK    {ds}/  ({found}/{len(ds_paths)} images found)")
        else:
            print(f"  FAIL  {ds}/  ({found}/{len(ds_paths)} images found, {missing_count} missing)")
            ok = False
            failures += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if ok:
        print("Result: ALL CHECKS PASSED")
    else:
        print(f"Result: {failures} CHECK(S) FAILED — see details above")
    print("=" * 60)

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the completeness of a SearchAD dataset directory.")
    parser.add_argument(
        "--searchad-dir",
        type=str,
        required=True,
        help="Path to the SearchAD directory to check.",
    )
    args = parser.parse_args()

    check_searchad_setup(searchad_dir=args.searchad_dir)
