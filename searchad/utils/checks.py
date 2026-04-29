import os
import warnings
from typing import Any


def check_prediction_labels(
    predictions: dict[str, list[str]],
    searchad_labels: list[str],
) -> int:
    """
    Checks that all expected labels are present in predictions and no unknown labels exist.

    Args:
        predictions: Mapping of SearchAD label to ranked image path list.
        searchad_labels: List of all valid SearchAD label names.

    Returns:
        Number of issues found.
    """
    failed_count = 0

    for expected_label in searchad_labels:
        if expected_label not in predictions:
            warnings.warn(
                f"Sanity Check Warning: Expected label '{expected_label}' is missing from the predictions file.",
            )
            failed_count += 1

    known_labels_set = set(searchad_labels)
    for actual_label in predictions:
        if actual_label not in known_labels_set:
            warnings.warn(
                f"Sanity Check Warning: Unexpected key '{actual_label}' found in the predictions file. "
                "This key is not a valid SearchAD label and might be a typo.",
            )
            failed_count += 1

    return failed_count


def check_prediction_dataset_prefixes(
    predictions: dict[str, list[str]],
    known_datasets: list[str],
) -> int:
    """
    Checks that all image path prefixes correspond to known dataset names.

    Args:
        predictions: Mapping of SearchAD label to ranked image path list.
        known_datasets: List of valid dataset directory names (top-level path components).

    Returns:
        Number of issues found.
    """
    failed_count = 0
    known_datasets_set = set(known_datasets)
    datasets_in_predictions: set[str] = set()

    for paths_list in predictions.values():
        for image_path in paths_list:
            normalized_path = os.path.normpath(image_path)
            path_components = [c for c in normalized_path.split(os.sep) if c]
            if path_components:
                datasets_in_predictions.add(path_components[0])

    missing_datasets = known_datasets_set - datasets_in_predictions
    wrong_datasets = datasets_in_predictions - known_datasets_set

    if missing_datasets:
        warnings.warn(
            f"Sanity Check Warning: The following expected dataset names were not found as path prefixes "
            f"in the predictions file: {missing_datasets}.",
        )
        failed_count += 1
    if wrong_datasets:
        warnings.warn(
            f"Sanity Check Warning: The following dataset names appear as path prefixes in the predictions "
            f"but are not valid SearchAD dataset names: {wrong_datasets}.",
        )
        failed_count += 1

    return failed_count


def check_prediction_list_lengths(
    predictions: dict[str, list[str]],
    total_images: int,
) -> int:
    """
    Checks that each label's ranked list contains exactly `total_images` entries.

    Args:
        predictions: Mapping of SearchAD label to ranked image path list.
        total_images: Expected number of images per list.

    Returns:
        Number of issues found.
    """
    failed_count = 0

    if total_images == 0:
        warnings.warn(
            "Sanity Check Warning: Mapping file contains no entries, skipping list length check.",
        )
        return 1

    for label, paths_list in predictions.items():
        if len(paths_list) != total_images:
            warnings.warn(
                f"Sanity Check Warning: For key '{label}', the list length ({len(paths_list)}) "
                f"does not match the total number of SearchAD test images ({total_images}).",
            )
            failed_count += 1

    return failed_count


def check_prediction_paths_in_mapping(
    predictions: dict[str, list[str]],
    imagepath_to_id: dict[str, int],
) -> int:
    """
    Checks that all image paths in predictions exist in the ID mapping.

    Args:
        predictions: Mapping of SearchAD label to ranked image path list.
        imagepath_to_id: Mapping from image path to integer ID.

    Returns:
        Number of issues found (0 or 1).
    """
    missing_paths_count = 0
    for paths_list in predictions.values():
        for image_path in paths_list:
            if image_path not in imagepath_to_id:
                missing_paths_count += 1

    if missing_paths_count > 0:
        warnings.warn(
            f"Sanity Check Warning: {missing_paths_count} image path(s) were not found in the "
            "mapping. These will be skipped during ID replacement.",
        )
        return 1

    return 0


def check_ground_truth_coverage(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
) -> int:
    """
    Checks that all ground truth images appear in the corresponding predicted lists.

    Args:
        predictions: Mapping of SearchAD label to ranked image path list.
        ground_truth: Mapping of SearchAD label to list of relevant image paths.

    Returns:
        Number of issues found (0 or 1).
    """
    missing_gt_count = 0
    for label, gt_paths in ground_truth.items():
        predicted_set = set(predictions.get(label, []))
        for gt_path in gt_paths:
            if gt_path not in predicted_set:
                missing_gt_count += 1

    if missing_gt_count > 0:
        warnings.warn(
            f"Sanity Check Warning: {missing_gt_count} ground truth image(s) across all labels are "
            "not present in the predictions. These will count as unranked and reduce your scores.",
        )
        return 1

    return 0


def check_submission_metadata(
    team_name: str,
    model_name: str,
    paper_code_affiliation: str,
    search_mode: str,
    searchad_train: str,
    default_queries: str,
) -> int:
    """
    Checks that all submission metadata fields are non-empty and have valid values.

    Args:
        team_name: Name of the submitting team.
        model_name: Name of the model.
        paper_code_affiliation: Link to paper/code or affiliation.
        search_mode: Search mode used (Language, Vision, or Multimodal).
        searchad_train: Whether SearchAD training set was used (Yes or No).
        default_queries: Whether default queries were used (Yes or No).

    Returns:
        Number of issues found.
    """
    failed_count = 0

    if not team_name.strip():
        warnings.warn("Metadata Sanity Check Warning: 'Team Name' cannot be empty.")
        failed_count += 1
    if not model_name.strip():
        warnings.warn("Metadata Sanity Check Warning: 'Model Name' cannot be empty.")
        failed_count += 1
    if not paper_code_affiliation.strip():
        warnings.warn("Metadata Sanity Check Warning: 'Paper/Code/Affiliation' cannot be empty.")
        failed_count += 1

    allowed_search_modes = {"language", "vision", "multimodal"}
    if search_mode.lower() not in allowed_search_modes:
        warnings.warn(
            f"Metadata Sanity Check Warning: 'Search Mode' must be one of {allowed_search_modes}. "
            f"Found '{search_mode}'.",
        )
        failed_count += 1

    allowed_yes_no = {"yes", "no"}
    if searchad_train.lower() not in allowed_yes_no:
        warnings.warn(
            f"Metadata Sanity Check Warning: 'SearchAD Train' must be 'Yes' or 'No'. " f"Found '{searchad_train}'.",
        )
        failed_count += 1
    if default_queries.lower() not in allowed_yes_no:
        warnings.warn(
            f"Metadata Sanity Check Warning: 'Default Queries' must be 'Yes' or 'No'. " f"Found '{default_queries}'.",
        )
        failed_count += 1

    return failed_count


def check_annotation_label_coverage(
    annotations: dict[str, list[dict[str, Any]]],
    searchad_labels: list[str],
) -> list[str]:
    """
    Returns labels from searchad_labels that have no annotations.

    Args:
        annotations: Annotation data in the form ``{image_path: [{"label": ..., ...}, ...]}``,
                     as loaded directly from a SearchAD annotation JSON file.
        searchad_labels: List of all expected SearchAD label names.

    Returns:
        List of label names that appear in searchad_labels but not in annotations.
    """
    labels_found = {ann["label"] for anns in annotations.values() for ann in anns if "label" in ann}
    return [lbl for lbl in searchad_labels if lbl not in labels_found]


def check_query_file_coverage(
    queries_dir: str,
    searchad_labels: list[str],
) -> tuple[list[str], int]:
    """
    Checks that every label has a corresponding ``.json`` query file in queries_dir.

    Args:
        queries_dir: Path to the directory containing per-label query JSON files.
        searchad_labels: List of all expected SearchAD label names.

    Returns:
        A tuple ``(missing_labels, total_query_files)`` where
        ``missing_labels`` is the list of labels with no query file and
        ``total_query_files`` is the total number of ``.json`` files found.
    """
    query_files = [f for f in os.listdir(queries_dir) if f.endswith(".json")]
    query_bases = {os.path.splitext(f)[0] for f in query_files}
    missing = [lbl for lbl in searchad_labels if lbl not in query_bases]
    return missing, len(query_files)
