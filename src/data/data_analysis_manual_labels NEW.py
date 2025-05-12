import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tkinter import Tk, filedialog

# === SETUP ===
segment_start, segment_end = 0, 35
default_label = 1
labels_map = {0: 'rest', 1: 'grip', 2: 'hold', 3: 'release'}
label_colors = {0: 'purple', 1: 'green', 2: 'brown', 3: 'pink'}

# === HARDCODED SAVE DIRECTORY ===
save_dir = r"C:\Users\Samsung\Desktop\Python_project_CDT406\Manual_labels"
os.makedirs(save_dir, exist_ok=True)

# === SELECT DATASET FOLDER ===
Tk().withdraw()  # Hide root Tk window
print("Select the folder containing your CSV files:")
dataset_dir = filedialog.askdirectory(title="Select dataset folder")
if not dataset_dir:
    print("No folder selected. Exiting.")
    exit()

# === LOAD FILE LIST ===
csv_files = glob.glob(os.path.join(dataset_dir, '**', '*.csv'), recursive=True)
if not csv_files:
    print("No CSV files found.")
    exit()

print("\nAvailable CSV files:")
for i, file in enumerate(csv_files):
    print(f"{i}: {os.path.basename(file)}")

file_index = int(input("\nSelect a file to label (enter number): "))
file_path = csv_files[file_index]

# === LOAD DATA ===
df = pd.read_csv(file_path, header=None, names=['time', 'measurement', 'label'])
df['label'] = df['label'].fillna(0).astype(int)
segment = df[(df['time'] >= segment_start) & (df['time'] <= segment_end)].copy().reset_index(drop=True)

# === STATE ===
current_label = [default_label]
history = []
status_message = ["Click two points to mark a label interval."]
click_points = []
saved_flag = [False]

# === DRAW FUNCTION ===
def redraw(*, reset_zoom=False):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.clear()
    if not reset_zoom:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    for label, color in label_colors.items():
        subset = segment[segment['label'] == label]
        ax.scatter(subset['time'], subset['measurement'], c=color, s=10, label=labels_map[label])

    ax.set_title(f"Label: {labels_map[current_label[0]]} (0–3 = change, u = undo, s = save, q = quit)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal")
    ax.legend()
    ax.grid(True)

    ax.text(0.01, 0.95, status_message[0], transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    guide = "\n".join([f"{k} = {v}" for k, v in labels_map.items()])
    ax.text(0.99, 0.95, guide, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8))

    fig.canvas.draw()

# === CLICK HANDLER ===
def onclick(event):
    if ax.get_navigate_mode() is not None:
        return
    if event.xdata is None:
        return
    click_points.append(event.xdata)
    if len(click_points) == 2:
        x1, x2 = sorted(click_points)
        idxs = segment[(segment['time'] >= x1) & (segment['time'] <= x2)].index
        if not idxs.empty:
            history.append((idxs.tolist(), segment.loc[idxs, 'label'].tolist()))
            segment.loc[idxs, 'label'] = current_label[0]
            status_message[0] = f"✓ Labeled {x1:.2f}–{x2:.2f} as {labels_map[current_label[0]]}"
        else:
            status_message[0] = "No points in selected interval."
        click_points.clear()
        redraw()
    else:
        status_message[0] = f"First point: {click_points[0]:.2f}"
        redraw()

# === KEY HANDLER ===
def onkey(event):
    if event.key in ['0', '1', '2', '3']:
        current_label[0] = int(event.key)
        status_message[0] = f"▶ Current label: {labels_map[current_label[0]]}"
        redraw()

    elif event.key == 'u' and history:
        idxs, old_labels = history.pop()
        segment.loc[idxs, 'label'] = old_labels
        status_message[0] = "↩ Undid last labeling"
        redraw()

    elif event.key == 's':
        if saved_flag[0]:
            status_message[0] = "Already saved! Press 'q' to quit."
            redraw()
            return

        # Apply labeled segment to the full data
        mask = (df['time'] >= segment_start) & (df['time'] <= segment_end)
        df.loc[mask, 'label'] = segment['label'].values

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        filename = f"{base_name}_manual_labeled.csv"
        file_path_out = os.path.join(save_dir, filename)
        df.to_csv(file_path_out, index=False)

        status_message[0] = f"Saved to: {file_path_out}"
        saved_flag[0] = True
        redraw()

    elif event.key == 'q':
        plt.close(fig)


# === LAUNCH PLOT ===
plt.close('all')

fig, ax = plt.subplots(figsize=(14, 6))
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)

redraw(reset_zoom=True)
plt.show()
