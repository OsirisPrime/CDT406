# src/utils/path_utils.py
from pathlib import Path
from typing import Iterable

def get_project_root() -> Path:
    """Return the path to the project root directory."""
    # Starting from this file's location, go up to project root
    return Path(__file__).resolve().parent.parent.parent


def get_raw_data_dir() -> Path:
    """Return the path to the raw data directory."""
    return get_project_root() / "data" / "raw"


def get_processed_data_dir() -> Path:
    """Return the path to the processed data directory."""
    return get_project_root() / "data" / "processed"


def get_processed_old_data_dir() -> Path:
    """Return the path to the processed old data directory."""
    return get_project_root() / "data" / "processed" / "relabeled_old_dataset"


def get_processed_fast_data_dir() -> Path:
    """Return the path to the processed fast data directory."""
    return get_project_root() / "data" / "processed" / "relabeled_fast_dataset"

def get_filtered_data_dir(ignore_folders: Iterable[str] = ()) -> list[Path]:
    """
    Return a list of subfolders in the processed data directory,
    optionally ignoring multiple specific subfolders.
    input: list_processed_subfolders(ignore_folders=["subject_05", "temp"])
    """
    base_dir = get_processed_data_dir()
    ignore_set = set(ignore_folders)
    return [folder for folder in base_dir.iterdir()
            if folder.is_dir() and folder.name not in ignore_set]

def get_logs_dir() -> Path:
    """Return the path to the logs directory."""
    return get_project_root() / "logs"


def get_models_dir() -> Path:
    """Return the path to the saved models directory."""
    return get_project_root() / "models"


