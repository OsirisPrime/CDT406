import time
import numpy as np
import Adafruit_BBIO.ADC as ADC
from tflite_interpreter import TFLiteModelInterpreter
from src.models.model_components.preprocessor import SignalPreprocessor

# === CONFIGURATION ===
EMG_CHANNELS = ["P9_40"]
SAMPLE_RATE_HZ = 5000
WINDOW_DURATION_MS = 0.2
STRIDE_DURATION_MS = 0.05
pre_processor_variant = 1

WINDOW_SIZE = int(WINDOW_DURATION_MS * SAMPLE_RATE_HZ)
STRIDE = int(STRIDE_DURATION_MS * SAMPLE_RATE_HZ)

CLASS_NAMES = ["Rest", "Grip", "Hold", "Release"]
TFLITE_MODEL_PATH = "LSTM_variant_1.tflite"

# === SETUP ===
ADC.setup()
model = TFLiteModelInterpreter(TFLITE_MODEL_PATH)

num_channels = len(EMG_CHANNELS)
data_buffer = []

print(f"Starting EMG inference with:")
print(f"- {num_channels} channels")
print(f"- {WINDOW_SIZE} sample window ({WINDOW_DURATION_MS} ms)")
print(f"- {STRIDE} sample stride ({STRIDE_DURATION_MS} ms)\n")

pre_processor = SignalPreprocessor(pre_processor_variant=pre_processor_variant,
                                               low_freq=20.0,
                                               high_freq=500.0,
                                               fs=5000.0,
                                               order=7)

try:
    while True:
        # --- 1. Read EMG sample ---
        sample = [ADC.read(pin) for pin in EMG_CHANNELS]
        data_buffer.append(sample)

        # --- 2. Check if enough samples for a new window ---
        if len(data_buffer) >= WINDOW_SIZE and (len(data_buffer) - WINDOW_SIZE) % STRIDE == 0:
            window = np.array(data_buffer[-WINDOW_SIZE:], dtype=np.float32)

            pre_processor.pre_process(window)
            input_data = np.expand_dims(window, axis=0)  # Shape: (1, 1000, num_channels)

            # --- 3. Inference ---
            output = model.predict(input_data)
            predicted_idx = int(np.argmax(output))
            predicted_label = CLASS_NAMES[predicted_idx]
            print(f"[{time.strftime('%H:%M:%S')}] Prediction: {predicted_label}")

        # --- 4. Optional: Limit buffer size ---
        max_buffer = WINDOW_SIZE + STRIDE
        if len(data_buffer) > max_buffer:
            data_buffer = data_buffer[-max_buffer:]

        # --- 5. Wait for next sample ---
        time.sleep(1.0 / SAMPLE_RATE_HZ)

except KeyboardInterrupt:
    print("\nStopped by user.")
