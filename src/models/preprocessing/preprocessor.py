from scipy import signal
import tensorflow as tf
import numpy as np
from scipy.signal import butter, sosfiltfilt

class SignalPreprocessor:
    def __init__(self, low_freq=20., high_freq=500., fs=5000, order=7):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = fs
        self.order = order
        self.sos = self.butter_bandpass()
        self.variance = 1.0  # Default, will be set in calibrate

    def butter_bandpass(self):
        nyq = 0.5 * self.fs
        low = self.low_freq / nyq
        high = self.high_freq / nyq
        sos = butter(self.order, [low, high], btype='band', output='sos')
        return sos

    def butter_bandpass_filter(self, data):
        y = sosfiltfilt(self.sos, data)
        return y

    def calibrate(self, raw_data):
        processed_signals = []
        for source, group in raw_data.groupby('source'):
            signal_array = group['measurement'].values
            processed = self.pre_process(signal_array)
            processed_signals.append(np.array(processed))
        all_processed = np.concatenate(processed_signals)
        self.variance = np.var(all_processed)

    def pre_process(self, x):
        # Bandpass filter
        x = self.butter_bandpass_filter(x)

        # Absolute value
        x = np.abs(x)

        # Moving average (window size 200, output same length)
        window_size = 200
        window = np.ones(window_size) / window_size
        x = np.convolve(x, window, mode='same')

        # Normalization
        x = (x - 0.0) / (np.sqrt(self.variance) + 1e-8)

        return x

    def batch_pre_process(self, X):
        """
        Apply pre_process to each sample in a batch.

        Parameters:
            X (np.ndarray): 2D array of shape (n_samples, window_length)

        Returns:
            np.ndarray: Processed array of same shape
        """
        return np.apply_along_axis(self.pre_process, 1, X)