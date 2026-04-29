import math
import warnings
from collections import defaultdict

import torch


def assign_category(query: str, categories: list[str]) -> str:
    """
    Assigns a category to a query based on its name.

    Args:
        query (str): The query string.
        categories (List[str]): List of valid SearchAD category names.

    Returns:
        str: The category the query belongs to.

    Raises:
        ValueError: If the query cannot be assigned to a known category.
    """
    query = query.lower()
    if "animal" in query and "sign" not in query:
        return "Animal"
    elif "human" in query or "person" in query:
        return "Human"
    elif "object" in query:
        return "Object"
    elif "rideable" in query:
        return "Rideable"
    elif "marking" in query:
        return "Marking"
    elif "scene" in query:
        return "Scene"
    elif "sign" in query:
        return "Sign"
    elif "trailer" in query:
        return "Trailer"
    elif "vehicle-construction" in query:
        return "Vehicle"
    elif "vehicle-duty" in query:
        return "Vehicle"
    elif "vehicle-special" in query:
        return "Vehicle"
    elif "vehicle" in query:
        return "Vehicle"
    else:
        for known_category in categories:
            if known_category.lower() in query:
                return known_category
        raise ValueError(
            f"The category of the search query '{query}' could not be assigned to one of the SearchAD categories!"
        )


def calculate_category_averages(
    data: dict[str, dict[str, float]], metric: str, categories: list[str]
) -> dict[str, float | None]:
    """
    Calculates the average of a specified metric for each category.

    Args:
        data (dict): A dictionary where keys are queries and values are dictionaries
                     containing metrics (e.g., {"MAP": 0.5, "R-Precision": 0.6}).
        metric (str): The name of the metric to average (e.g., "MAP", "R-Precision", "P@5").
        categories (List[str]): List of valid SearchAD category names.

    Returns:
        dict: A dictionary where keys are categories and values are their average metric scores.
              Values are float or None if no data for the category.
    """
    category_metrics: dict[str, list[float]] = defaultdict(list)
    for query, metrics in data.items():
        metric_value = metrics.get(metric)
        if metric_value is not None and not math.isnan(metric_value):
            try:
                category = assign_category(query, categories)
                category_metrics[category].append(metric_value)
            except ValueError as e:
                warnings.warn(f"{e}. Skipping query '{query}' for category average calculation.")
                continue

    category_averages: dict[str, float | None] = {}
    for category, metrics_list in category_metrics.items():
        avg_metric = sum(metrics_list) / len(metrics_list) if metrics_list else None
        category_averages[category] = avg_metric
    return category_averages


def mean_average_precision(sorted_target: torch.Tensor, top_k: int = 100) -> torch.Tensor:
    """
    Calculates the Mean Average Precision (MAP) for a given sorted target tensor.

    Args:
        sorted_target (torch.Tensor): A 1D tensor where 1 indicates a relevant item
                                      and 0 indicates an irrelevant item, sorted by relevance score.
        top_k (int): The maximum number of items to consider.

    Returns:
        torch.Tensor: The MAP score. Returns 0.0 if no relevant items are present.
    """
    num_relevant_positives = torch.sum(sorted_target).int()
    if num_relevant_positives == 0:
        return torch.tensor(0.0)

    effective_len = min(top_k, len(sorted_target))
    truncated_target = sorted_target[:effective_len]

    cumulative_relevant_count = torch.cumsum(truncated_target, dim=0)
    denominators = torch.arange(1, effective_len + 1, device=truncated_target.device, dtype=torch.float32)
    precision_at_k = cumulative_relevant_count / denominators

    relevant_indices_in_truncated = torch.nonzero(truncated_target).squeeze(-1)

    # Sum precision_at_k only at those relevant positions
    acc_avg_precision = torch.sum(precision_at_k[relevant_indices_in_truncated])

    return acc_avg_precision / num_relevant_positives


def mean_rprecision(sorted_target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean R-Precision for a given sorted target tensor.

    Args:
        sorted_target (torch.Tensor): A 1D tensor where 1 indicates a relevant item
                                      and 0 indicates an irrelevant item, sorted by relevance score.

    Returns:
        torch.Tensor: The R-Precision score. Returns 0.0 if no relevant items are present.
    """
    num_relevant = torch.sum(sorted_target).int()
    if num_relevant == 0:
        return torch.tensor(0.0)
    # R-Precision is precision at the point where the number of retrieved items
    # equals the number of relevant items in the result set.
    r = torch.sum(sorted_target[:num_relevant])
    return r / num_relevant


def precision_at_k(sorted_target: torch.Tensor, k: int) -> torch.Tensor:
    """
    Calculates Precision@K.

    Args:
        sorted_target (torch.Tensor): A 1D tensor where 1 indicates a relevant item
                                      and 0 indicates an irrelevant item, sorted by relevance score.
        k (int): The number of top items to consider.

    Returns:
        torch.Tensor: The Precision@K score.
    """
    effective_k = min(k, len(sorted_target))
    if effective_k == 0:
        return torch.tensor(0.0)
    relevant_in_top_k = torch.sum(sorted_target[:effective_k])
    return relevant_in_top_k / effective_k
