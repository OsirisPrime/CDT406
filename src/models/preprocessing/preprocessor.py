from scipy import signal
import tensorflow as tf
import numpy as np

def calibrate_pre_processor(raw_data, fs=5000, low_freq=20, high_freq=500):
    nyquist = 0.5 * 5000
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')

    processed_signals = []

    for source, group in raw_data.groupby('source'):
        signal_array = group['measurement'].values
        processed = pre_processor_layer(signal_array, b, a)
        processed_signals.append(np.array(processed))

    all_processed = np.concatenate(processed_signals)

    variance = np.var(all_processed)

    return a, b, variance

def pre_processor_layer(x, b, a, variance=1):

    # Bandpass 20-500 Hz (also test using signal.lfilter)
    x = signal.filtfilt(b, a, x)

    # Absolute value
    x = tf.abs(x)

    # Moving average 10 samples
    x = tf.signal.frame(x, frame_length=200, frame_step=1)
    x = tf.reduce_mean(x, axis=1)

    x = tf.keras.layers.Normalization(mean=0, variance=variance)(x)

    return x