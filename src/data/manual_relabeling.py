import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- Setup ------------
input_folder = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\test_data"
output_folder = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\test_output"

files = sorted(os.listdir(input_folder))
if not files:
    print("No files found in the input folder.")
    exit()

file_name = files[0]
input_path = os.path.join(input_folder, file_name)
output_path = os.path.join(output_folder, f"edited_{file_name}")

# ----------- Load and Process Data ------------
data = pd.read_csv(input_path, header=None)
data.columns = ['time', 'voltage', 'label']

# Store the original data to plot before editing
original_data = data.copy()

def detect_transitions(df):
    df['prev_label'] = df['label'].shift(1)
    trans = df[df['label'] != df['prev_label']].copy()
    trans = trans[pd.notna(trans['prev_label'])]

    # Adding transition duration
    trans['duration'] = trans['time'].diff().shift(-1)
    return trans

# ----------- Plot Original Data (Before Editing) ------------
plt.figure(figsize=(10, 6))
for label in original_data['label'].unique():
    label_data = original_data[original_data['label'] == label]
    plt.scatter(label_data['time'], label_data['voltage'], label=f'Label {int(label)}', alpha=0.6, s=10)

plt.title('Original EMG Signal Scatter Plot (Before Editing)')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (mV)')
plt.legend(title='Labels')
plt.grid(True)
plt.tight_layout()
plt.show()

while True:
    # Recalculate transitions after every edit
    transitions = detect_transitions(data)
    transition_indices = transitions.index.tolist()

    print("\nDetected transitions:")
    for i, idx in enumerate(transition_indices):
        row = data.loc[idx]
        duration = transitions.loc[idx, 'duration']
        print(f"{i}: Transition from {int(row['prev_label'])} → {int(row['label'])} at time {row['time']} (Duration: {duration:.2f}s)")

    idx_input = input("\nEnter index of the transition to edit (or press Enter to finish): ")
    if not idx_input.strip():
        break  # Exit edit loop

    if not idx_input.strip().isdigit():
        print("Invalid input. Please enter a valid index.")
        continue

    idx = int(idx_input)
    if idx not in range(len(transition_indices)):  # Check against the new reindexed range
        print("Index not found in transition list.")
        continue

    old_time = data.at[transition_indices[idx], 'time']
    old_label = data.at[transition_indices[idx], 'label']
    prev_label = data.at[transition_indices[idx], 'prev_label']

    new_time_input = input(f"Enter new timestamp for transition from {int(prev_label)} → {int(old_label)} (was at time {old_time}): ")
    try:
        new_time = float(new_time_input)
        new_idx = (np.abs(data['time'] - new_time)).idxmin()

        # Find region to relabel
        current_pos = transition_indices.index(transition_indices[idx])
        start_idx = transition_indices[current_pos - 1] + 1 if current_pos > 0 else 0
        end_idx = transition_indices[current_pos + 1] if current_pos + 1 < len(transition_indices) else len(data)

        # Relabel
        data.loc[start_idx:new_idx - 1, 'label'] = prev_label
        data.loc[new_idx:end_idx - 1, 'label'] = old_label
        print(f"✔ Transition updated at index {new_idx} (actual time {data.at[new_idx, 'time']})")

    except ValueError:
        print("Invalid timestamp input. Skipping...")

# ----------- Clean and Save ------------
data.drop(columns='prev_label', inplace=True)
os.makedirs(output_folder, exist_ok=True)
data.to_csv(output_path, index=False, header=False)
print(f"\n✅ Modified file saved to: {output_path}")

# ----------- Plot Updated Data (After Editing) ------------
plt.figure(figsize=(10, 6))
for label in data['label'].unique():
    label_data = data[data['label'] == label]
    plt.scatter(label_data['time'], label_data['voltage'], label=f'Label {int(label)}', alpha=0.6, s=10)

plt.title('Updated EMG Signal Scatter Plot (After Editing)')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (mV)')
plt.legend(title='Labels')
plt.grid(True)
plt.tight_layout()
plt.show()
