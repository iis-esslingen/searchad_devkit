import json
import os
import warnings
from typing import Any


def load_json(path: str | os.PathLike) -> Any:
    """Load and return a JSON file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not valid JSON.
    """
    path = str(path)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in {path}: {e}") from e


def save_json(data: Any, path: str | os.PathLike, indent: int = 4) -> None:
    """Serialize *data* to a JSON file at *path*."""
    with open(str(path), "w") as f:
        json.dump(data, f, indent=indent)


def subdataset_for_path(img_path: str, subdatasets: list[str]) -> str | None:
    """Return the subdataset name that is a prefix of *img_path*, or None."""
    for ds in subdatasets:
        if img_path.startswith(ds + "/") or img_path.startswith(ds + os.sep):
            return ds
    return None


def load_query_files(queries_dir: str | os.PathLike) -> dict:
    all_query_data: dict = {}
    if not os.path.exists(queries_dir):
        warnings.warn(f"Queries directory not found at {queries_dir}")
        return all_query_data

    print(f"Loading query files from: {queries_dir}")
    for filename in sorted(os.listdir(queries_dir)):
        if filename.endswith(".json"):
            file_path = os.path.join(queries_dir, filename)
            try:
                data = load_json(file_path)
                filename_base = os.path.splitext(filename)[0]
                all_query_data[filename_base] = data
                print(f"  Loaded: {filename}")
            except Exception as e:
                warnings.warn(f"Could not load {file_path}: {e}")
    return all_query_data
