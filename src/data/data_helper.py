import pandas as pd
from src.utils.path_utils import get_processed_data_dir
import glob


def get_raw_data_as_dataframe():
    expected_columns = ['time', 'measurement', 'label']
    csv_files = glob.glob(str(get_processed_data_dir()) + "/**/*.csv", recursive=True)
    dfs = []
    for file in csv_files:
        with open(file, 'r') as f:
            first_line = f.readline().strip().split(',')
        if first_line == expected_columns:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(file, header=None, names=expected_columns)
        df['source'] = file
        dfs.append(df)
    raw_data = pd.concat(dfs, ignore_index=True)
    raw_data = raw_data.sort_values(['source', 'time']).reset_index(drop=True)
    return raw_data

import numpy as np
import pandas as pd
from collections import Counter

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
