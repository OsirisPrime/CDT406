import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.io import RawArray
from mne import create_info
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


# Parameters
sampling_rate = 1000
window_size = 200
overlap = 0.5
l_freq = 20.0
h_freq = 450.0


def preprocess_emg(data, fs):
    info = create_info(ch_names=["EMG"], sfreq=fs, ch_types=["misc"])
    raw_data = RawArray(data[np.newaxis, :], info, verbose=False)
    raw_data.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", picks="misc", verbose=False)
    filtered = raw_data.get_data()[0]
    scaler = StandardScaler()
    normalized = scaler.fit_transform(filtered.reshape(-1, 1)).flatten()
    return normalized


def create_segments(data, window_size, overlap):
    segments = []
    step = int(window_size * (1 - overlap))
    for i in range(0, len(data) - window_size, step):
        segments.append(data[i:i + window_size])
    return np.array(segments)


def plot_first_four_all_together(raw_signals, preprocessed_signals, titles):
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for i in range(4):
        axs[i].plot(raw_signals[i] - np.mean(raw_signals[i]), label='Raw (Zero-mean)', alpha=0.5)
        axs[i].plot(preprocessed_signals[i], label='Preprocessed (Norm)', linewidth=1)
        axs[i].set_title(titles[i])
        axs[i].legend(loc='upper right')
        axs[i].set_ylabel('Amplitude')
        axs[i].set_xlim(0, 13000)
        axs[i].set_ylim(-15, 15)
    axs[-1].set_xlabel('Samples')
    plt.tight_layout()
    plt.show()


# TD Feature Extraction
def extract_td_features(data):
    # Calculate all 10 features in one go for the given data window
    features = [
        np.mean(np.abs(data)),  # MAV: Mean Absolute Value
        np.sqrt(np.mean(data**2)),  # RMS: Root Mean Square
        np.count_nonzero(np.diff(np.sign(data)) != 0),  # ZC: Zero Crossing
        np.count_nonzero(np.diff(np.sign(np.diff(data))) != 0),  # SSC: Slope Sign Change
        np.sum(np.abs(np.diff(data))),  # WL: Waveform Length
        np.var(data),  # Variance
        np.mean((data - np.mean(data))**3) / (np.std(data)**3),  # Skewness
        np.mean((data - np.mean(data))**4) / (np.std(data)**4),  # Kurtosis
        np.var(data),  # H_A: Hjorth Activity (similar to variance)
        np.sqrt(np.var(np.diff(data)) / np.var(data))  # H_M: Hjorth Mobility (difference ratio)
    ]
    return features

def process_emg_data(base_dir, output_dir, visualize_only=False, full_run=False):
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    all_data = []  # To hold the processed EMG data with features
    raw_signals = []
    preprocessed_signals = []
    titles = []

    segment_counter = 0  # To keep global segment index
    subjects = range(1, 29)

    with tqdm(total=28, desc="Subjects") as pbar:
        for subject in subjects:
            for cycle in range(1, 4):
                cycle_folder = os.path.join(base_dir, f"subject #{subject}", f"cycle #{cycle}")
                if not os.path.exists(cycle_folder):
                    continue
                files = sorted([f for f in os.listdir(cycle_folder) if os.path.isfile(os.path.join(cycle_folder, f))])

                for i, file in enumerate(files):  # i used for sensor index
                    file_path = os.path.join(cycle_folder, file)
                    try:
                        data = np.loadtxt(file_path, delimiter=',')
                        if len(data) < window_size:
                            continue

                        raw = data.copy()
                        processed = preprocess_emg(data, sampling_rate)

                        if visualize_only or full_run:
                            if len(raw_signals) < 4:
                                raw_signals.append(raw)
                                preprocessed_signals.append(processed)
                                titles.append(f"Subject {subject}, Sensor {i + 1}, File {file}")

                        if not visualize_only or full_run:
                            segments = create_segments(processed, window_size, overlap)
                            step = int(window_size * (1 - overlap)) if overlap else window_size

                            for segment in segments:
                                td_features = extract_td_features(segment)
                                all_data.append(td_features)

                    except Exception as e:
                        print(f"Failed to process {file}: {e}")
            pbar.update(1)

    if (not visualize_only or full_run) and all_data:
        all_data = np.vstack(all_data)  # Concatenate all features into one array
        os.makedirs(output_dir, exist_ok=True)

        with tqdm(total=2, desc="Saving") as saver_bar:
            np.savez_compressed(os.path.join(output_dir, 'X_train.npz'), data=all_data)  # Save only the features
            saver_bar.update(1)

            df_data = pd.DataFrame(all_data)
            df_data.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)  # Save as CSV as well
            saver_bar.update(1)

        print("✅ Data processed and saved.")
    elif (not visualize_only or full_run) and not all_data:
        print("⚠️ No data was processed.")

    if (visualize_only or full_run) and raw_signals:
        plot_first_four_all_together(raw_signals, preprocessed_signals, titles)
    elif (visualize_only or full_run) and not raw_signals:
        print("⚠️ No signals to visualize.")




def main():
    print("Choose an option:")
    print("1: Visualize first 4 EMG signals (raw & preprocessed)")
    print("2: Process and save EMG data")
    print("3: Both visualize and process data")
    choice = input("Enter your choice (1/2/3): ")

    base_dir = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\M6F1O1 Dataset"
    output_dir = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\FIN data"

    if choice == "1":
        process_emg_data(base_dir, output_dir, visualize_only=True)
    elif choice == "2":
        process_emg_data(base_dir, output_dir, visualize_only=False)
    elif choice == "3":
        process_emg_data(base_dir, output_dir, full_run=True)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()