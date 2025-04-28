import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.path_utils import get_processed_data_dir
from src.models.ANN.simpleANN import SimpleANN
from src.utils.model_utils import save_best_model

def main():
    """
    Main function to load, preprocess, split, and train the ANN model.

    Steps:
    1. Load data from an .npz file.
    2. Adjust label values by decreasing each by 1.
    3. Determine the unique number of label classes.
    4. Initialize the SimpleANN model.
    5. Split the data into training and validation sets.
    6. Train the model.
    """
    # Build the file path for the processed data file.
    path = str(get_processed_data_dir()) + "/emg_data.npz"

    # Load the data file, retrieving data and labels.
    npz_file = np.load(path)
    data = npz_file['data']
    labels = npz_file['labels']

    # Adjust label range from 1-28 to 0-27.
    labels = labels - 1

    # Determine the number of unique classes in the labels.
    num_unique_labels = np.unique(labels).shape[0]

    # Initialize the SimpleANN model with the input shape and number of classes.
    model = SimpleANN(input_shape=data.shape[1], num_classes=num_unique_labels)

    # Split the data into 80% training and 20% validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    # Train the model using the training and validation datasets.
    model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    save_best_model(model, "simple_ann")

if __name__ == "__main__":
    main()