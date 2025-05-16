import time
import numpy as np
import Adafruit_BBIO.ADC as ADC
import tflite_runtime.interpreter as tflite
from scipy.signal import butter, lfilter

# === CONFIGURATION ===
EMG_CHANNELS = ["P9_40"]
SAMPLE_RATE_HZ = 5000           # 5000 samples per second
WINDOW_DURATION_MS = 0.2        # window = 200 ms (in seconds)
STRIDE_DURATION_MS = 0.05       # shift = 50 ms (25% overlap)

WINDOW_SIZE = int(WINDOW_DURATION_MS * SAMPLE_RATE_HZ)  # = 1000 samples
STRIDE = int(STRIDE_DURATION_MS * SAMPLE_RATE_HZ)       # = 250 samples

CLASS_NAMES = ["Rest", "Grip", "Hold", "Release"]
TFLITE_MODEL_PATH = "model.tflite"

# === FILTER SETUP ===
def butter_bandpass(lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

b, a = butter_bandpass(20, 500, SAMPLE_RATE_HZ, order=7)

def moving_average(signal, window_size=20):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def preprocess_emg(signal_window):
    """
    signal_window: np.array shape (WINDOW_SIZE, num_channels)
    Returns filtered, rectified, smoothed, normalized signal.
    """
    processed = []
    for ch in range(signal_window.shape[1]):
        raw = signal_window[:, ch]
        filtered = lfilter(b, a, raw)
        rectified = np.abs(filtered)
        smoothed = moving_average(rectified, window_size=20)
        norm = (smoothed - np.min(smoothed)) / (np.ptp(smoothed) + 1e-6)
        processed.append(norm)
    return np.stack(processed, axis=1)

# === SETUP ===
ADC.setup()

interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

num_channels = len(EMG_CHANNELS)
data_buffer = []

print(f"Starting EMG inference with:")
print(f"- {num_channels} channels")
print(f"- {WINDOW_SIZE} sample window ({WINDOW_DURATION_MS} ms)")
print(f"- {STRIDE} sample stride ({STRIDE_DURATION_MS} ms)\n")

try:
    while True:
        # --- 1. Read EMG sample ---
        sample = [ADC.read(pin) for pin in EMG_CHANNELS]
        data_buffer.append(sample)

        # --- 2. Check if enough samples for a new window ---
        if len(data_buffer) >= WINDOW_SIZE and (len(data_buffer) - WINDOW_SIZE) % STRIDE == 0:
            window = np.array(data_buffer[-WINDOW_SIZE:], dtype=np.float32)
            processed = preprocess_emg(window)
            input_data = np.expand_dims(processed, axis=0)  # Shape: (1, 20, num_channels)

            # --- 3. Inference ---
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            predicted_idx = int(np.argmax(output))
            predicted_label = CLASS_NAMES[predicted_idx]
            print(f"[{time.strftime('%H:%M:%S')}] Prediction: {predicted_label}")

        # --- 4. Optional: Limit buffer size to avoid memory growth ---
        max_buffer = WINDOW_SIZE + STRIDE
        if len(data_buffer) > max_buffer:
            data_buffer = data_buffer[-max_buffer:]

        # --- 5. Wait for next sample ---
        time.sleep(1.0 / SAMPLE_RATE_HZ)

except KeyboardInterrupt:
    print("\nStopped by user.")
