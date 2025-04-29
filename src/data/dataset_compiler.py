import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from mne.filter import filter_data
from scipy.stats import zscore
from scipy.stats import skew, kurtosis
from scipy.signal import correlate


# Global Variables
sample_rate = 1000      # The sample rate of the signal (Hz)
low_freq = 20           # Bandpass filter low frequency (Hz)
high_freq = 450         # Bandpass filter high frequency (Hz)
window_size = 1000      # The number of samples per segment/window
overlap_ratio = 0.5     # Overlap ratio (0 to 1)
discard_segment = True  # Keep only full segments (True or False)


# Segments the EMG data into overlapping windows.
def segment_emg_data(data, window_size, step_size):
    """
    :param data: The EMG signal to segment.
    :param window_size: The number of samples in the window.
    :param step_size: The number of samples to skip between windows.
    :return: A list of segments. If "discard_segment=True", only full segments
             will be kept. Otherwise, smaller segments will be included.
    """

    num_samples = len(data)
    num_segments = (num_samples - window_size) // step_size + 1  # Number of overlapping segments

    segments = []
    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + window_size

        # Ensure the end index doesn't exceed the signal length
        if end_idx > num_samples:
            end_idx = num_samples
        segment = data[start_idx:end_idx]

        # If throw_segment is True, discard the segment if it's not full (i.e., shorter than window_size)
        if discard_segment and len(segment) < window_size:
            continue  # Discard this segment

        segments.append(segment)
    return segments


# Extract 10 time-domain features from the EMG segment.
def extract_td_features(segment):
    mean = np.mean(segment)                         # 1. Mean
    rms = np.sqrt(np.mean(segment ** 2))            # 2. Root Mean Square (RMS)
    p2p = np.max(segment) - np.min(segment)         # 3. Peak-to-Peak (P2P)
    variance = np.var(segment)                      # 4. Variance
    skewness = skew(segment)                        # 5. Skewness
    kurt = kurtosis(segment)                        # 6. Kurtosis
    zcr = np.sum(np.diff(np.sign(segment)) != 0)    # 7. Zero Crossing Rate (ZCR)
    sem = np.mean(np.abs(segment))                  # 8. Signal Envelope Mean (SEM)
    autocorr = correlate(segment, segment, mode='full')[len(segment) - 1]   # 9. Autocorrelation (Lag 1)

    # 10. Entropy
    entropy = -np.sum(
        np.log(np.abs(np.histogram(segment, bins=10)[0] + 1e-10)) * np.histogram(segment, bins=10)[0] / len(segment))

    features = [mean, rms, p2p, variance, skewness, kurt, zcr, sem, autocorr, entropy]
    return features

# Apply bandpass filter and normalize the segment
def filter_and_normalize(segment):
    segment = filter_data(segment, sfreq=sample_rate, l_freq=low_freq, h_freq=high_freq, verbose=False) # Apply bandpass filter
    segment = zscore(segment)   # Normalize using z-score
    return segment


def read_and_segment_emg(base_dir, output_dir, window, overlap):
    all_rows = []
    subjects = range(1, 29)

    # Calculate the step size for overlapping segments
    step_size = int(window * (1 - overlap))

    with tqdm(total=28, desc="Subjects") as pbar:
        for subject in subjects:
            for cycle in range(1, 4):
                cycle_folder = os.path.join(base_dir, f"subject #{subject}", f"cycle #{cycle}")
                if not os.path.exists(cycle_folder):
                    continue
                files = sorted([f for f in os.listdir(cycle_folder) if os.path.isfile(os.path.join(cycle_folder, f))])

                for file in files:
                    file_path = os.path.join(cycle_folder, file)
                    try:
                        data = np.loadtxt(file_path, delimiter=',')  # shape: (N,)

                        # Segment the data
                        segments = segment_emg_data(data, window, step_size)

                        for i, segment in enumerate(segments):
                            segment = filter_and_normalize(segment) # Apply filter and normalization
                            features = extract_td_features(segment) # Extract time-domain features

                            # Assign label based on the second
                            segment_time = (i * step_size) // 1000  # To determine the time in seconds for the label
                            if segment_time < 5:
                                label = 0  # Rest
                            elif segment_time < 11:
                                label = 1  # Hold
                            else:
                                label = 2  # Release

                            # Append features and label
                            all_rows.append(features + [label])

                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")

            pbar.update(1)

    if all_rows:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(all_rows)

        # Columns are time-domain features + label
        feature_columns = [f"feature_{i}" for i in range(1, 11)]
        columns = feature_columns + ["label"]
        df.columns = columns

        df.to_csv(os.path.join(output_dir, 'S3M6F1O1_features_dataset.csv'), index=False)
        print("✅ Segmented, filtered, normalized EMG data with TD features saved to CSV.")
    else:
        print("⚠️ No data found to save.")


def main():
    base_dir = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\S3M6F1O1 Dataset"
    output_dir = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\Dataset"

    read_and_segment_emg(base_dir, output_dir, window_size, overlap_ratio)


if __name__ == "__main__":
    main()
