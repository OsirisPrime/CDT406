import csv

from src.utils.path_utils import get_logs_dir


def setup_logging(model_name):
    """Set up local logging."""
    logs_dir = get_logs_dir() / model_name / "local_logging"

    logs_dir.mkdir(parents=True, exist_ok=True)

    return logs_dir


def log_metrics(model_name, data: dict, file_name=None):
    with open(get_logs_dir() / model_name / "local_logging" / (file_name if file_name else model_name), mode="w",
              newline="") as file:
        writer = csv.writer(file)

        # Write data
        for key, value in data.items():
            writer.writerow([key, value])


def load_metrics(model_name, file_name=None):
    with open(get_logs_dir() / model_name / "local_logging" / (file_name if file_name else model_name),
              mode="r") as file:
        reader = csv.reader(file)

        dictionary = dict(reader)

        for row in reader:
            print(row)

    return dictionary
