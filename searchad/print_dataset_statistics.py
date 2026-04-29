import argparse
import os
import warnings
from collections import defaultdict
from collections.abc import Sequence

from searchad.config.config import SEARCHAD_DATASETS, SEARCHAD_LABELS
from searchad.utils.io import load_json, subdataset_for_path


def _count_image_level(data: dict, label_filter: set) -> dict:
    """Return {label: number_of_images_containing_label}."""
    counts: dict = defaultdict(int)
    for annotations in data.values():
        labels_in_image = {ann["label"] for ann in annotations if ann["label"] in label_filter}
        for label in labels_in_image:
            counts[label] += 1
    return dict(counts)


def _count_object_level(data: dict, label_filter: set) -> dict:
    """Return {label: total_bounding_box_count}."""
    counts: dict = defaultdict(int)
    for annotations in data.values():
        for ann in annotations:
            label = ann["label"]
            if label in label_filter:
                counts[label] += 1
    return dict(counts)


def _count_image_level_by_subdataset(data: dict, label_filter: set, subdatasets: list) -> tuple[dict, dict]:
    """Return ({subdataset: {label: count}}, {subdataset: image_count})."""
    counts: dict = {ds: defaultdict(int) for ds in subdatasets}
    img_counts: dict = defaultdict(int)
    for img_path, annotations in data.items():
        ds = subdataset_for_path(img_path, subdatasets)
        if ds is None:
            continue
        img_counts[ds] += 1
        labels_in_image = {ann["label"] for ann in annotations if ann["label"] in label_filter}
        for label in labels_in_image:
            counts[ds][label] += 1
    return {ds: dict(v) for ds, v in counts.items()}, dict(img_counts)


def _count_object_level_by_subdataset(data: dict, label_filter: set, subdatasets: list) -> tuple[dict, dict]:
    """Return ({subdataset: {label: count}}, {subdataset: image_count})."""
    counts: dict = {ds: defaultdict(int) for ds in subdatasets}
    img_counts: dict = defaultdict(int)
    for img_path, annotations in data.items():
        ds = subdataset_for_path(img_path, subdatasets)
        if ds is None:
            continue
        img_counts[ds] += 1
        for ann in annotations:
            label = ann["label"]
            if label in label_filter:
                counts[ds][label] += 1
    return {ds: dict(v) for ds, v in counts.items()}, dict(img_counts)


def _print_table(
    rows: list,
    col_names: list,
    label_col_name: str = "Label",
    output_file: str | None = None,
) -> None:
    """Print a right-aligned plain-text table (no external dependencies).

    Args:
        rows: List of dicts mapping column name -> cell value.
        col_names: Ordered list of column names (excluding the label column).
        label_col_name: Name of the row-label column (left-aligned).
        output_file: If given, also write the table to this file.
    """
    # Compute column widths
    widths = {label_col_name: len(label_col_name)}
    for col in col_names:
        widths[col] = len(str(col))
    for row in rows:
        widths[label_col_name] = max(widths[label_col_name], len(str(row.get(label_col_name, ""))))
        for col in col_names:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    def _fmt_row(row: dict) -> str:
        line = f"  {str(row.get(label_col_name, '')):.<{widths[label_col_name]}}"
        for col in col_names:
            line += f"  {str(row.get(col, '')):>{widths[col]}}"
        return line

    header = f"  {label_col_name:<{widths[label_col_name]}}"
    for col in col_names:
        header += f"  {str(col):>{widths[col]}}"
    sep = "-" * len(header)

    lines = [sep, header, sep] + [_fmt_row(r) for r in rows] + [sep]
    table = "\n".join(lines)

    print(table)
    if output_file:
        with open(output_file, "w") as f:
            f.write(table + "\n")


def print_dataset_statistics(
    searchad_dir: str | os.PathLike,
    splits: Sequence[str] = ["train", "val"],
    level: str = "object",
    statistics_type: str = "absolute",
    output_dir: str | os.PathLike | None = None,
    by_subdataset: bool = False,
) -> None:
    """Print label distribution statistics for the SearchAD dataset.

    Reads the unified annotation JSON files and counts how often each of the
    90 SearchAD labels appears, broken down by split.  Results are printed as a
    plain-text table (no pandas required) and optionally saved to a .txt file.

    Args:
        searchad_dir: Path to the SearchAD directory containing
            ``searchad_annotations_{split}.json`` files.
        splits: Dataset splits to process.  Supported values: ``"train"``,
            ``"val"``.
        level: ``"object"`` counts total bounding-box annotations per label;
            ``"image"`` counts the number of images containing each label.
        statistics_type: ``"absolute"`` for raw counts; ``"relative"`` for
            each split's share of the label's total count (fractions that sum
            to 1 across splits).
        output_dir: If given, the table is saved as a ``.txt`` file in this
            directory.
        by_subdataset: If True, also print (and optionally save) a second table
            with per-subdataset counts aggregated over all requested splits.
            Subdatasets with zero annotations across all labels are excluded
            from the table automatically.
    """
    searchad_dir = os.path.abspath(searchad_dir)
    label_filter = set(SEARCHAD_LABELS)

    subdatasets = list(SEARCHAD_DATASETS)

    # ── Load data ─────────────────────────────────────────────────────────────
    counts: dict = {}  # counts[split][label] = int
    image_counts: dict = {}  # image_counts[split] = int

    # Subdataset accumulators (aggregated over all splits)
    sub_counts: dict = {ds: defaultdict(int) for ds in subdatasets}  # {ds: {label: int}}
    sub_img_counts: dict = defaultdict(int)  # {ds: int}

    for split in splits:
        ann_file = os.path.join(searchad_dir, f"searchad_annotations_{split}.json")
        if not os.path.exists(ann_file):
            warnings.warn(f"No annotations found for split '{split}'. Skipping.")
            continue
        data = load_json(ann_file)
        if not data:
            warnings.warn(f"Annotations file for split '{split}' is empty. Skipping.")
            continue
        image_counts[split] = len(data)
        if level == "image":
            counts[split] = _count_image_level(data, label_filter)
            if by_subdataset:
                ds_counts, ds_imgs = _count_image_level_by_subdataset(data, label_filter, subdatasets)
        else:
            counts[split] = _count_object_level(data, label_filter)
            if by_subdataset:
                ds_counts, ds_imgs = _count_object_level_by_subdataset(data, label_filter, subdatasets)
        if by_subdataset:
            for ds in subdatasets:
                for lbl, cnt in ds_counts[ds].items():
                    sub_counts[ds][lbl] += cnt
            for ds, cnt in ds_imgs.items():
                sub_img_counts[ds] += cnt

    available_splits = list(counts.keys())
    if not available_splits:
        print("No annotation data found. Exiting.")
        return

    # ── Sort labels by total count (ascending, rarest first) ─────────────────
    sorted_labels = sorted(
        SEARCHAD_LABELS,
        key=lambda lbl: sum(counts[s].get(lbl, 0) for s in available_splits),
    )

    col_names = available_splits + ["Total"]

    # ── Build label rows ──────────────────────────────────────────────────────
    rows = []
    for label in sorted_labels:
        split_vals = {s: counts[s].get(label, 0) for s in available_splits}
        total = sum(split_vals.values())
        row: dict = {"Label": label}
        if statistics_type == "relative":
            for s in available_splits:
                row[s] = f"{split_vals[s] / total:.2f}" if total > 0 else "0.00"
            row["Total"] = "1.00" if total > 0 else "0.00"
        else:
            for s in available_splits:
                row[s] = split_vals[s]
            row["Total"] = total
        rows.append(row)

    # ── Summary rows (always absolute) ───────────────────────────────────────
    rows.append(dict.fromkeys(["Label"] + col_names, "---"))

    total_ann_row: dict = {"Label": f"Total annotations ({level}-level)"}
    for s in available_splits:
        total_ann_row[s] = sum(counts[s].get(lbl, 0) for lbl in sorted_labels)
    total_ann_row["Total"] = sum(total_ann_row[s] for s in available_splits)
    rows.append(total_ann_row)

    classes_row: dict = {"Label": "Classes with annotations > 0"}
    for s in available_splits:
        classes_row[s] = sum(1 for lbl in sorted_labels if counts[s].get(lbl, 0) > 0)
    classes_row["Total"] = sum(1 for lbl in sorted_labels if any(counts[s].get(lbl, 0) > 0 for s in available_splits))
    rows.append(classes_row)

    img_row: dict = {"Label": "Image count"}
    for s in available_splits:
        img_row[s] = image_counts.get(s, 0)
    img_row["Total"] = sum(image_counts.get(s, 0) for s in available_splits)
    rows.append(img_row)

    # ── Output ────────────────────────────────────────────────────────────────
    title = (
        f"SearchAD Label Statistics"
        f"  |  level={level}"
        f"  |  type={statistics_type}"
        f"  |  splits={', '.join(available_splits)}"
    )
    print(f"\n{title}")

    output_file = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"label_statistics_{level}_{statistics_type}_{'_'.join(available_splits)}.txt"
        output_file = os.path.join(str(output_dir), fname)

    _print_table(rows, col_names, output_file=output_file)

    if level == "image":
        print(
            "Note: 'Total annotations (image-level)' counts images multiple times " "if they contain multiple labels."
        )
    if output_file:
        print(f"Table saved to: {output_file}")

    # ── Per-subdataset table ──────────────────────────────────────────────────
    if not by_subdataset:
        return

    # Drop subdatasets that have zero annotations across all labels
    active_subdatasets = [ds for ds in subdatasets if any(sub_counts[ds].get(lbl, 0) > 0 for lbl in sorted_labels)]

    sub_col_names = active_subdatasets + ["Total"]
    sub_rows = []
    for label in sorted_labels:
        ds_vals = {ds: sub_counts[ds].get(label, 0) for ds in active_subdatasets}
        total = sum(ds_vals.values())
        row: dict = {"Label": label}
        if statistics_type == "relative":
            for ds in active_subdatasets:
                row[ds] = f"{ds_vals[ds] / total:.2f}" if total > 0 else "0.00"
            row["Total"] = "1.00" if total > 0 else "0.00"
        else:
            for ds in active_subdatasets:
                row[ds] = ds_vals[ds]
            row["Total"] = total
        sub_rows.append(row)

    # Summary rows
    sub_rows.append(dict.fromkeys(["Label"] + sub_col_names, "---"))

    sub_total_ann_row: dict = {"Label": f"Total annotations ({level}-level)"}
    for ds in active_subdatasets:
        sub_total_ann_row[ds] = sum(sub_counts[ds].get(lbl, 0) for lbl in sorted_labels)
    sub_total_ann_row["Total"] = sum(sub_total_ann_row[ds] for ds in active_subdatasets)
    sub_rows.append(sub_total_ann_row)

    sub_classes_row: dict = {"Label": "Classes with annotations > 0"}
    for ds in active_subdatasets:
        sub_classes_row[ds] = sum(1 for lbl in sorted_labels if sub_counts[ds].get(lbl, 0) > 0)
    sub_classes_row["Total"] = sum(
        1 for lbl in sorted_labels if any(sub_counts[ds].get(lbl, 0) > 0 for ds in active_subdatasets)
    )
    sub_rows.append(sub_classes_row)

    sub_img_row: dict = {"Label": "Image count"}
    for ds in active_subdatasets:
        sub_img_row[ds] = sub_img_counts.get(ds, 0)
    sub_img_row["Total"] = sum(sub_img_counts.get(ds, 0) for ds in active_subdatasets)
    sub_rows.append(sub_img_row)

    sub_title = (
        f"SearchAD Label Statistics — by subdataset"
        f"  |  level={level}"
        f"  |  type={statistics_type}"
        f"  |  splits={', '.join(available_splits)}"
    )
    print(f"\n{sub_title}")

    sub_output_file = None
    if output_dir:
        sub_fname = f"label_statistics_{level}_{statistics_type}" f"_{'_'.join(available_splits)}_by_subdataset.txt"
        sub_output_file = os.path.join(str(output_dir), sub_fname)

    _print_table(sub_rows, sub_col_names, output_file=sub_output_file)

    if level == "image":
        print(
            "Note: 'Total annotations (image-level)' counts images multiple times " "if they contain multiple labels."
        )
    if sub_output_file:
        print(f"Table saved to: {sub_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print label distribution statistics for the SearchAD dataset.")
    parser.add_argument(
        "--searchad-dir",
        type=str,
        required=True,
        help="Path to the SearchAD directory.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="Dataset splits to process (default: train val).",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="object",
        choices=["image", "object"],
        help="'object' counts bounding-box annotations per label; "
        "'image' counts images containing each label. Default: object.",
    )
    parser.add_argument(
        "--statistics-type",
        type=str,
        default="absolute",
        choices=["absolute", "relative"],
        help="'absolute' for raw counts, 'relative' for fraction of total. Default: absolute.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional: directory to save the statistics table as a .txt file.",
    )
    parser.add_argument(
        "--by-subdataset",
        action="store_true",
        default=False,
        help="Also print per-subdataset statistics (subdatasets with all-zero counts are excluded).",
    )

    args = parser.parse_args()
    warnings.formatwarning = lambda message, *args, **kwargs: f"WARNING: {message}\n"

    print_dataset_statistics(
        searchad_dir=args.searchad_dir,
        splits=tuple(args.splits),
        level=args.level,
        statistics_type=args.statistics_type,
        output_dir=args.output_dir,
        by_subdataset=args.by_subdataset,
    )
