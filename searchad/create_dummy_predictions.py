import argparse
import os
import random

from searchad.config.config import SEARCHAD_LABELS
from searchad.utils.io import load_json, save_json


def create_dummy_predictions(
    searchad_dir: str | os.PathLike,
    predictions_file: str | os.PathLike,
    split: str,
    seed: int = 42,
) -> None:
    """
    Creates a random dummy prediction file for the specified split.

    For val and train, image paths are taken from the bounding-box annotation files.
    For test, image paths are taken from the test ID mapping file.
    Each label receives an independently shuffled copy of all image paths.

    Args:
        searchad_dir: Path to the SearchAD directory.
        predictions_file: Path where the output predictions JSON file will be saved.
        split: Dataset split to generate predictions for: ``train``, ``val``, or ``test``.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(predictions_file)), exist_ok=True)

    if split in ("val", "train"):
        ann_path = os.path.join(searchad_dir, f"searchad_annotations_{split}.json")
        print(f"Loading {split} annotations from: {ann_path}")
        annotations = load_json(ann_path)
        all_image_paths = list(annotations.keys())
    else:  # test
        test_mapping_path = os.path.join(searchad_dir, "searchad_test_mapping_id_to_imagepath.json")
        print(f"Loading test mapping from: {test_mapping_path}")
        test_mapping = load_json(test_mapping_path)
        all_image_paths = list(test_mapping.values())

    predictions = {}
    for label in SEARCHAD_LABELS:
        shuffled = all_image_paths.copy()
        random.shuffle(shuffled)
        predictions[label] = shuffled

    output_path = predictions_file
    save_json(predictions, output_path)
    print(f"Saved {len(SEARCHAD_LABELS)} labels × {len(all_image_paths)} images → {output_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a random dummy prediction file for a specified SearchAD split."
    )
    parser.add_argument(
        "--searchad-dir",
        type=str,
        required=True,
        help="Path to the SearchAD directory.",
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        required=True,
        help="Path where the output predictions JSON file will be saved.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Dataset split to generate predictions for: 'train', 'val', or 'test'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    create_dummy_predictions(
        searchad_dir=args.searchad_dir,
        predictions_file=args.predictions_file,
        split=args.split,
        seed=args.seed,
    )
