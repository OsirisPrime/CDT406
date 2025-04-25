import csv
from src.utils.path_utils import get_logs_dir

def setup_logging(model_name):
    """
    Set up a directory for local logging for a specific model.

    This function creates a directory structure for storing logs related to a given model.
    The directory will be created under the logs directory, with the structure:
    logs/<model_name>/local_logging.

    Args:
        model_name (str): The name of the model for which logging is being set up.

    Returns:
        Path: The path to the created logging directory.
    """
    logs_dir = get_logs_dir() / model_name / "local_logging"

    # Create the directory and any necessary parent directories if they don't exist
    logs_dir.mkdir(parents=True, exist_ok=True)

    return logs_dir


def log_metrics(model_name, data: dict, file_name=None):
    """
    Log metrics to a CSV file for a specific model.

    This function writes key-value pairs from the provided dictionary to a CSV file.
    The file is stored in the logging directory for the given model.

    Args:
        model_name (str): The name of the model for which metrics are being logged.
        data (dict): A dictionary containing the metrics to log. Keys are metric names, and values are their values.
        file_name (str, optional): The name of the CSV file. If not provided, the model name is used as the file name.

    Returns:
        None
    """
    # Determine the file path for the CSV file
    file_path = get_logs_dir() / model_name / "local_logging" / (file_name if file_name else model_name)

    # Open the file in write mode and write the metrics
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write each key-value pair from the dictionary to the CSV file
        for key, value in data.items():
            writer.writerow([key, value])


def load_metrics(model_name, file_name=None):
    """
    Load metrics from a CSV file for a specific model.

    This function reads key-value pairs from a CSV file and returns them as a dictionary.
    The file is read from the logging directory for the given model.

    Args:
        model_name (str): The name of the model for which metrics are being loaded.
        file_name (str, optional): The name of the CSV file. If not provided, the model name is used as the file name.

    Returns:
        dict: A dictionary containing the loaded metrics. Keys are metric names, and values are their values.
    """
    # Determine the file path for the CSV file
    file_path = get_logs_dir() / model_name / "local_logging" / (file_name if file_name else model_name)

    # Open the file in read mode and read the metrics
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)

        # Convert the CSV rows into a dictionary
        dictionary = dict(reader)

        # Print each row (for debugging or logging purposes)
        for row in reader:
            print(row)

    return dictionary