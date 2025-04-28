import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.path_utils import get_processed_data_dir
from src.models.KNN.KNN import SimpleKNN

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
    path = str(get_processed_data_dir()) + "/S3M6F1O1_dataset.csv"

    # Load the data file
    data = pd.read_csv(path)

    # Get X data and y data
    y_data = data['label'].values
    X_data = data.drop(columns=['label']).values

    # Transform to numpy array
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # Initialize the SimpleKNN model
    knn = SimpleKNN(n_neighbors=5)

    # Split the data into 80% training and 20% validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    # Train the model using the training.
    knn.train(X_train, y_train)

    # Evaluate the model on validation data.
    accuracy = knn.evaluate(X_val, y_val)

    # Print the accuracy.
    print("Validation Accuracy:", accuracy)

if __name__ == "__main__":
    main()