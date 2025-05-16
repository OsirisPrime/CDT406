import tensorflow as tf
from tensorflow.keras import layers, models

from src.models.preprocessing.stft_layer import STFTLayer

class LSTM_STFT:
    """
    LSTM is a recurrent neural network model for classification.

    The model expects to receive a raw window of data. Before feeding the window into
    the LSTM layer, the data is processed by applying a bandpass filter and normalization.

    Usage:
        1. Instantiate with the window input shape and number of classes:
           model = LSTM(input_shape=<window_length, features>, num_classes=<number_of_classes>)

        2. Train the model using:
           model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)

        3. Evaluate the model on test data:
           loss, accuracy = model.evaluate(X_test, y_test)
    """
    model_name = "LSTM_STFT"

    def __init__(self,
                 input_shape,
                 num_classes,
                 learning_rate=1e-3,
                 optimizer='adam',
                 normalization='none',
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 act_dense='tanh',
                 act_lstm='tanh',
                 units_dense1=32,
                 units_lstm=16,
                 stft_frame_length=64,
                 stft_frame_step=32):
        """
        Initialize the LSTM model.

        Parameters:
            input_shape (int): The number of time steps in the input window.
            num_classes (int): Number of classes for classification.
        """
        self.frame_length = stft_frame_length
        self.frame_step = stft_frame_step

        def stft_layer(x):
            # x shape: (batch, time)
            stft = tf.signal.stft(x, frame_length=self.frame_length, frame_step=self.frame_step)
            spectrogram = tf.abs(stft)
            return spectrogram

        # Build the network
        net = []
        net.append(layers.Input(shape=(input_shape,)))
        net.append(layers.Reshape((1, -1)))

        net.append(STFTLayer(
            frame_length=stft_frame_length,
            frame_step=stft_frame_step,
            name='stft'
        ))

        net.append(layers.Reshape((1, -1)))

        # optional normalisation
        if normalization == 'batch':
            net.append(layers.BatchNormalization())
        elif normalization == 'layer':
            net.append(layers.LayerNormalization())

        net.append(layers.LSTM(units_lstm,
                               activation=act_lstm,
                               unroll=True,
                               recurrent_dropout=recurrent_dropout))

        net.append(layers.Dense(units_dense1, activation=act_dense))

        if dropout > 0:
            net.append(layers.Dropout(dropout))

        net.append(layers.Dense(num_classes, activation='softmax'))

        self.model = models.Sequential(net)

        # pick optimiser
        opt = self._get_optimizer(optimizer, learning_rate)

        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.F1Score(average='macro')]
        )

    @staticmethod
    def _get_optimizer(name, lr):
        """Return an optimiser instance given name & learning-rate."""
        name = name.lower()
        if name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=lr)
        if name == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=lr)
        if name == 'nadam':
            return tf.keras.optimizers.Nadam(learning_rate=lr)
        raise ValueError(f"Unknown optimiser: {name}")

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, verbose=2):
        """
        Train the LSTM model.

        Parameters:
            X_train (array-like): Raw training windows.
            y_train (array-like): Training labels.
            X_val (array-like): Raw validation windows.
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
            X_test (array-like): Raw test windows.
            y_test (array-like): Test labels.

        Returns:
            Tuple containing loss and accuracy.
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