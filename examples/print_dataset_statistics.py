from searchad.print_dataset_statistics import print_dataset_statistics

SEARCHAD_DIR = "/path/to/searchad"  # Update this path to your SearchAD directory
SPLITS = ["train", "val"]  # "train", "val", or both
LEVEL = "object"  # "object" (bounding boxes per label) or "image" (images containing label)
STATISTICS_TYPE = "absolute"  # "absolute" for raw counts, "relative" for fraction of total
OUTPUT_DIR = "/path/to/dataset_statistics"  # Update this path to your output directory (set to None to skip saving)
BY_SUBDATASET = True  # Also print per-subdataset statistics

print_dataset_statistics(
    searchad_dir=SEARCHAD_DIR,
    splits=SPLITS,
    level=LEVEL,
    statistics_type=STATISTICS_TYPE,
    output_dir=OUTPUT_DIR,
    by_subdataset=BY_SUBDATASET,
)
