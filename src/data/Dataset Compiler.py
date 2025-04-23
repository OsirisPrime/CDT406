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
window_size = 500
overlap = 0.5
l_freq = 2.0
h_freq = 150.0


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


def process_emg_data(base_dir, output_dir, visualize_only=False, full_run=False):
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    all_data = []
    all_labels = []
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
                                titles.append(f"Subject {subject}, Sensor {i+1}, File {file}")

                        if not visualize_only or full_run:
                            segments = create_segments(processed, window_size, overlap)
                            step = int(window_size * (1 - overlap)) if overlap else window_size

                            gesture_labels = []
                            for j in range(len(segments)):
                                global_index = segment_counter * step
                                if global_index <= 6000:
                                    gesture_labels.append(0)
                                elif global_index <= 12000:
                                    gesture_labels.append(1)
                                else:
                                    gesture_labels.append(2)
                                segment_counter += 1

                            all_data.append(segments)
                            all_labels.append(np.array(gesture_labels))

                    except Exception as e:
                        print(f"Failed to process {file}: {e}")
            pbar.update(1)

    if (not visualize_only or full_run) and all_data:
        all_data = np.vstack(all_data)
        all_labels = np.concatenate(all_labels)
        os.makedirs(output_dir, exist_ok=True)

        with tqdm(total=2, desc="Saving") as saver_bar:
            np.savez_compressed(os.path.join(output_dir, 'emg_data.npz'), data=all_data, labels=all_labels)
            saver_bar.update(1)

            df_data = pd.DataFrame(all_data.reshape(all_data.shape[0], -1))
            df_labels = pd.DataFrame(all_labels, columns=["Label"])
            df = pd.concat([df_data, df_labels], axis=1)
            df.to_csv(os.path.join(output_dir, 'emg_data.csv'), index=False)
            saver_bar.update(1)

        print("✅ Data processed and saved.")
        unique, counts = np.unique(all_labels, return_counts=True)
        print("Label distribution:", dict(zip(unique, counts)))

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

    base_dir = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\Dataset Training"
    output_dir = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\FIN data 2"

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
