# RNN will get a window, unprocessed.
# It then will normalize and "filter" the data.
# Once the data is filtered it is reshaped to fit with the LSTM layer.
# The RNN then iterates over the sequence of data inside the window.
# It only requires one window to train and make predictions.

import numpy as np
from tensorflow.keras import layers, models
from scipy.signal import butter, filtfilt


class RNN:
    """
    RNN is a recurrent neural network model for classification.

    The model expects to receive a raw window of data. Before feeding the window into
    the RNN layer, the data is processed by applying a bandpass filter and normalization.

    Usage:
        1. Instantiate with the window input shape and number of classes:
           model = RNN(input_shape=<window_length, features>, num_classes=<number_of_classes>)

        2. Train the model using:
           model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)

        3. Evaluate the model on test data:
           loss, accuracy = model.evaluate(X_test, y_test)
    """

    def __init__(self, input_shape, num_classes):
        """
        Initialize the RNN model.

        Parameters:
            input_shape (tuple): The shape of the input window (time_steps, features).
            num_classes (int): Number of classes for classification.
        """
        # Build the sequential model.
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            # layers.Lambda(lambda x: self._preprocess_data(x), name="preprocessing"), # Might add back later.
            layers.Dense(32, activation='relu'), # Based on the architecture of a paper (not sure which one yet)
            layers.LSTM(64, unroll=True),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def _bandpass_filter(self, data, lowcut=20, highcut=450, fs=1000, order=4):
        """
        Apply a bandpass filter to the data.

        Parameters:
            data (1D array): Data to be filtered.
            lowcut (float): Low frequency cutoff.
            highcut (float): High frequency cutoff.
            fs (float): Sampling frequency.
            order (int): Order of the filter.

        Returns:
            Filtered data.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def _normalize(self, data):
        """
        Normalize the data using z-score normalization.

        Parameters:
            data (1D array): Data to be normalized.

        Returns:
            Normalized data.
        """
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-7)

    def _preprocess_data(self, x):
        """
        Preprocess the input tensor. For each window in the batch, each feature
        is filtered and normalized. This can be done using tf.py_function to keep
        the preprocessing in the model graph.

        Parameters:
            x (tf.Tensor): Input tensor of shape (batch, time_steps, features).

        Returns:
            Preprocessed tensor.
        """
        import tensorflow as tf

        def preprocess_window(window):
            # window: (time_steps, features) numpy array
            proc_window = np.empty_like(window)
            # Process each feature column separately
            for i in range(window.shape[-1]):
                col = window[:, i]
                filtered = self._bandpass_filter(col)
                proc_window[:, i] = self._normalize(filtered)
            return proc_window.astype(np.float32)

        # Use tf.py_function to apply the numpy processing to each sample.
        processed = tf.py_function(func=lambda batch: np.array(
            [preprocess_window(window) for window in batch]), inp=[x], Tout=tf.float32)
        processed.set_shape(x.shape)
        return processed

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, verbose=2):
        """
        Train the RNN model.

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