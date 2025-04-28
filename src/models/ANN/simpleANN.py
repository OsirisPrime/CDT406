from tensorflow.keras import layers, models

class SimpleANN:
    """
    SimpleANN is a simple artificial neural network model for classification.

    Usage:
        1. Instantiate the model with the desired input shape and number of classes:
           model = SimpleANN(input_shape=<number_of_features>, num_classes=<number_of_classes>)

        2. Train the model using:
           model.train()

        3. Evaluate the model on test data:
           loss, accuracy = model.evaluate(X_test=<test_features>, y_test=<test_labels>)
           print("Test Accuracy:", accuracy)
    """

    def __init__(self, input_shape, num_classes):
        """
        Initialize the SimpleANN model.

        Parameters:
            input_shape (int): The number of features in the input data.
            num_classes (int): The number of classes for classification.
        """
        self.model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, verbose=2):
        """
        Train the SimpleANN model.

        Parameters:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
            X_val (array-like): Validation features.
            y_val (array-like): Validation labels.
            epochs (int, optional): Number of training epochs. Default is 10.
            batch_size (int, optional): Size of each training batch. Default is 32.
            verbose (int, optional): Verbosity mode. Default is 2.
        """
        self.model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=verbose)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Parameters:
            X_test (array-like): Test features.
            y_test (array-like): Test labels.

        Returns:
            Tuple containing loss and accuracy.
        """
        return self.model.evaluate(X_test, y_test, verbose=2)