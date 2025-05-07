import pandas as pd
from src.utils.path_utils import get_raw_data_dir
import glob

def get_raw_data_as_dataframe():
    """
    Loads and concatenates all CSV files from the raw data directory into a single DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - time: Timestamp or time value from the CSV.
            - measurement: Measurement value from the CSV.
            - label: Label value from the CSV.
            - source: File path of the CSV file from which the row originated.
    """
    # List all csv files
    csv_files = glob.glob(str(get_raw_data_dir()) + "/Dataset_O1/**/*.csv", recursive=True)

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file, header=None, names=['time', 'measurement', 'label'])
        df['source'] = file  # Track file origin
        dfs.append(df)

    raw_data = pd.concat(dfs, ignore_index=True)
    raw_data = raw_data.sort_values(['source', 'time']).reset_index(drop=True)

    return raw_data

