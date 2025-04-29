import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_and_segment_emg(base_dir, output_dir):
    all_rows = []

    subjects = range(1, 29)

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
                        data = np.loadtxt(file_path, delimiter=',')

                        num_samples = len(data)
                        num_seconds = num_samples // 1000  # Full 1-second segments only

                        for i in range(num_seconds):
                            segment = data[i * 1000:(i + 1) * 1000]

                            # Assign label based on the second
                            if i < 5:
                                label = 0  # Rest
                            elif i < 11:
                                label = 1  # Hold
                            else:
                                label = 2  # Release

                            all_rows.append(np.append(segment, label))  # Append segment + label

                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")

            pbar.update(1)

    if all_rows:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(all_rows)

        # Last column is the label
        feature_columns = [f"sample_{i}" for i in range(1000)]
        columns = feature_columns + ["label"]
        df.columns = columns

        df.to_csv(os.path.join(output_dir, 'S3M6F1O1_dataset.csv'), index=False)
        print("✅ Segmented EMG data saved to CSV.")
    else:
        print("⚠️ No data found to save.")


def main():
    base_dir = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\S3M6F1O1 Dataset"
    output_dir = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\Dataset"

    read_and_segment_emg(base_dir, output_dir)


if __name__ == "__main__":
    main()
