from tensorflow.keras import layers, models
import tensorflow as tf

from src.models.model_components.stft_layer import STFTLayer

class ANN_STFT:
    """
    ANN is a feed-forward neural network model for classification.

    The model consists of 3 dense layers with customizable sizes (default 32-16-32).

    Usage:
        1. Instantiate with the input shape and number of classes:
           model = ANN(input_shape=<features>, num_classes=<number_of_classes>)

        2. Train the model using:
           model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)

        3. Evaluate the model on test data:
           loss, accuracy = model.evaluate(X_test, y_test)
    """
    model_name = "ANN"

    def __init__(self,
                 input_shape,
                 num_classes,
                 learning_rate=1e-3,
                 optimizer='adam',
                 normalization='none',
                 dropout=0.0,
                 activation='tanh',
                 units_dense1=32,
                 units_dense2=16,
                 units_dense3=32,
                 stft_frame_length=64,
                 stft_frame_step=32):

        """
        Parameters
        ----------
        input_shape        : int      – number of features per example
        num_classes        : int      – #classes
        learning_rate      : float
        optimizer          : {'adam','rmsprop','nadam'}
        normalization      : {'none','batch','layer'}
        dropout            : float    – 0 … 0.5; used in Dropout layers
        activation         : {'relu','tanh','sigmoid'}
        units_dense1       : int      – units in first dense layer
        units_dense2       : int      – units in second dense layer
        units_dense3       : int      – units in third dense layer
        """

        # Build the network
        net = []
        net.append(layers.Input(shape=(input_shape,)))

        net.append(STFTLayer(
            frame_length=stft_frame_length,
            frame_step=stft_frame_step,
            name='stft'
        ))

        net.append(layers.Flatten())

        # First dense layer
        net.append(layers.Dense(units_dense1, activation=activation))

        if dropout > 0:
            net.append(layers.Dropout(dropout))

        # Optional normalization after first layer
        if normalization == 'batch':
            net.append(layers.BatchNormalization())
        elif normalization == 'layer':
            net.append(layers.LayerNormalization())

        # Second dense layer
        net.append(layers.Dense(units_dense2, activation=activation))

        if dropout > 0:
            net.append(layers.Dropout(dropout))

        # Optional normalization after second layer
        if normalization == 'batch':
            net.append(layers.BatchNormalization())
        elif normalization == 'layer':
            net.append(layers.LayerNormalization())

        # Third dense layer
        net.append(layers.Dense(units_dense3, activation=activation))

        if dropout > 0:
            net.append(layers.Dropout(dropout))

        # Output layer
        net.append(layers.Dense(num_classes, activation='softmax'))

        self.model = models.Sequential(net)

        # Pick optimizer
        opt = self._get_optimizer(optimizer, learning_rate)

        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.F1Score(average='macro')]
        )

    @staticmethod
    def _get_optimizer(name, lr):
        """Return an optimizer instance given name & learning-rate."""
        name = name.lower()
        if name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=lr)
        if name == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=lr)
        if name == 'nadam':
            return tf.keras.optimizers.Nadam(learning_rate=lr)
        raise ValueError(f"Unknown optimizer: {name}")

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, verbose=2):
        """
        Train the ANN model.

        Parameters:
            X_train (array-like): Training feature vectors.
            y_train (array-like): Training labels.
            X_val (array-like): Validation feature vectors.
            y_val (array-like): Validation labels.
            epochs (int, optional): Number of epochs. Default is 10.
            batch_size (int, optional): Batch size for training. Default is 32.
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
            X_test (array-like): Test feature vectors.
            y_test (array-like): Test labels.

        Returns:
            Tuple containing loss and F1 score.
        """
        return self.model.evaluate(X_test, y_test, verbose=2)

    def save(self, filepath, *args, **kwargs):
        """
        Save the model to the specified filepath.

        Parameters:
            filepath (str): File path where the model will be saved.
        """
        return self.model.save(filepath, *args, **kwargs)

    def load_weights(self, filepath, *args, **kwargs):
        """
        Load model weights from the specified filepath.

        Parameters:
            filepath (str): File path from which to load model weights.
        """
        return self.model.load_weights(filepath, *args, **kwargs)

    def get_model(self):
        """
        Get the underlying Keras model.

        Returns:
            The Keras model.
        """
        return self.model

    def get_model_name(self):
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self.model_name