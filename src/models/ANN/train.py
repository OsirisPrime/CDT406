import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.path_utils import get_processed_data_dir
from src.models.ANN.simpleANN import SimpleANN

def main():
    path = str(get_processed_data_dir()) + "/emg_data.npz"

    npz_file = np.load(path)
    data = npz_file['data']
    labels = npz_file['labels']

    # Convert label range from 1-28 to 0-27 (temporary fix)
    labels = labels - 1

    # Get the number of labels
    num_unique_labels = np.unique(labels).shape[0]

    model = SimpleANN(input_shape=data.shape[1], num_classes=num_unique_labels)

    # Split: 80% for training, 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

if __name__ == "__main__":
    main()