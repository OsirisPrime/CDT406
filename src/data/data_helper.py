from src.utils.path_utils import get_processed_data_dir

import glob
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import os


def get_raw_data_as_dataframe(validation_subjects=(1, 2)):
    """
    Load every CSV in  relabeled_old_dataset/ and split it into
    training and validation sets based on the *top-level* folder
    name (1 … 9).

    Parameters
    ----------
    validation_subjects : iterable of int, default (1, 2)
        The subject IDs (folder names) that should be placed
        in the validation split.

    Returns
    -------
    raw_train_data : pd.DataFrame
    raw_val_data   : pd.DataFrame
    """
    expected_columns = ["time", "measurement", "label"]

    # base_dir = Path(get_processed_data_dir()) / "relabeled_old_dataset"
    base_dir = Path(get_processed_data_dir()) / "new_standard_labeled_measurements"
    csv_files = base_dir.rglob("*.csv")

    train_frames, val_frames = [], []

    for file in csv_files:
        # -----------------------------------------------------------
        # Figure out which subject (folder 1…9) this file comes from
        # -----------------------------------------------------------
        try:
            subject_id = int(file.relative_to(base_dir).parts[0])
        except (ValueError, IndexError):
            # Skip any unexpected directory structure
            continue

        # -----------------------------------------------------------
        # Read the CSV – handle optional header row
        # -----------------------------------------------------------
        with open(file, "r") as f:
            first_line = f.readline().strip().split(",")

        if first_line == expected_columns:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(file, header=None, names=expected_columns)

        df["source"] = str(file)  # keep full path for traceability

        # -----------------------------------------------------------
        # Add to the proper split
        # -----------------------------------------------------------
        if subject_id in validation_subjects:
            val_frames.append(df)
        else:
            train_frames.append(df)

    # ---------------------------------------------------------------
    # Concatenate & sort to replicate the original raw_data format
    # ---------------------------------------------------------------
    raw_train_data = (
        pd.concat(train_frames, ignore_index=True)
          .sort_values(["source", "time"])
          .reset_index(drop=True)
    )

    raw_val_data = (
        pd.concat(val_frames, ignore_index=True)
          .sort_values(["source", "time"])
          .reset_index(drop=True)
    )

    return raw_train_data, raw_val_data

def segement_data(raw_data, window_length, overlap):
    """
    Segments the data into overlapping windows, only keeping windows from a single source.

    Args:
        raw_data (pd.DataFrame): Input data with columns ['time', 'measurement', 'label', 'source']
        window_length (int): Number of samples per window
        overlap (int): Number of samples to overlap between windows

    Returns:
        pd.DataFrame: Each row contains ['window_data', 'label', 'source']
    """

    step = window_length - overlap
    segments = []
    for start in range(0, len(raw_data) - window_length + 1, step):
        window = raw_data.iloc[start:start + window_length]
        sources = window['source'].values
        # Only keep if all sources are the same
        if not np.all(sources == sources[0]):
            continue
        measurements = window['measurement'].values
        labels = window['label'].values
        majority_label = Counter(labels).most_common(1)[0][0]
        segments.append({'window_data': measurements, 'label': majority_label, 'source': sources[0]})
    return pd.DataFrame(segments)
