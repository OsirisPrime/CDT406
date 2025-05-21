from scipy import signal
import tensorflow as tf
import numpy as np
from scipy.signal import butter, sosfiltfilt

class SignalPreprocessor:
    def __init__(self,
                 pre_processor_variant = 1,
                 low_freq=20.,
                 high_freq=500.,
                 fs=5000, order=7,
                 variance=1.0,
                 down_sample=True
                 ):

        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = fs
        self.order = order
        self.sos = self.butter_bandpass()
        self.variance = variance  # Default is 1, will be set in calibrate
        self.pre_processor_variant = pre_processor_variant
        self.down_sample = down_sample

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
        return self.variance

    def pre_process(self, x):
        # Bandpass filter
        x = self.butter_bandpass_filter(x)

        # Down sample from 5000 Hz to 1000 Hz
        if self.down_sample:
            x = signal.resample(x, int(1000 * 1000 / 5000))

        # Absolute value
        if self.pre_processor_variant == 1 or self.pre_processor_variant == 2:
            x = np.abs(x)

        # Moving average (window size 50, output same length)
        if self.pre_processor_variant == 1:
            if self.down_sample:
                window_size = 10
            else:
                window_size = 50
            window = np.ones(window_size) / window_size
            x = np.convolve(x, window, mode='same')

        if self.pre_processor_variant == 1:
            self.variance = np.float64(0.0035935865169082993)
        elif self.pre_processor_variant == 2:
            self.variance = np.float64(0.004820291414145334)
        else:
            self.variance = np.float64(0.006385202366156638)

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