import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class SimpleKNN:
    """
    SimpleKNN is a simple k-nearest neighbors classifier for classification.

    Usage:
        1. Instantiate the model with desired hyperparameters:
           model = SimpleKNN(n_neighbors=5)

        2. Train the model using:
           model.train(X_train=<training_features>, y_train=<training_labels>)

        3. Evaluate the model on test data:
           accuracy = model.evaluate(X_test=<test_features>, y_test=<test_labels>)
           print("Test Accuracy:", accuracy)
    """

    def __init__(self, n_neighbors=5):
        """
        Initialize the SimpleKNN model.

        Parameters:
            n_neighbors (int, optional): Number of neighbors to use. Default is 5.
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        """
        Train the SimpleKNN model.

        Note:
            KNeighborsClassifier training is just fitting the model to the data.

        Parameters:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Parameters:
            X_test (array-like): Test features.
            y_test (array-like): Test labels.

        Returns:
            Accuracy.
        """
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        # KNN does not compute a loss by default; return 0.0 as a placeholder.
        return accuracy

    def save(self, filepath):
        """
        Save the SimpleKNN model to the specified file using pickle.

        Parameters:
            filepath (str): Path to the file where the model will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load_weights(self, filepath):
        """
        Load model weights from the specified file using pickle.

        Parameters:
            filepath (str): Path to the file from which the model weights will be loaded.
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)