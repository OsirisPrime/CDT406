import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.utils.path_utils import get_raw_data_dir
from src.models.LSTM_STFT.LSTM_STFT import LSTM_STFT
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
    path = str(get_raw_data_dir()) + "/S3M6F1O1_dataset.csv"

    # Load the data file
    data = pd.read_csv(path)

    # Determine the number of unique classes in the labels.
    num_unique_labels = data['label'].nunique()

    # Get X data and y data
    y_data = data['label'].values
    X_data = data.drop(columns=['label']).values

    # Transform to numpy array
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    y_data = tf.keras.utils.to_categorical(y_data, num_classes=num_unique_labels)

    model = LSTM_STFT(input_shape=X_data.shape[1], num_classes=num_unique_labels)

    # Split the data into 80% training and 20% validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    # Train the model using the training and validation datasets.
    model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, epochs=10)

    save_best_model(model, "LSTM_STFT_test")

if __name__ == "__main__":
    main()