import os
import glob
import tensorflow as tf
from src.utils.path_utils import get_models_dir

# Define the custom STFT layer
class stft_layer(tf.keras.layers.Layer):
    def __init__(self, frame_length, frame_step):
        super(stft_layer, self).__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step

    def call(self, x):
        stft = tf.signal.stft(x, frame_length=self.frame_length, frame_step=self.frame_step)
        spectrogram = tf.abs(stft)
        return spectrogram

# Get list of all .keras models in the models directory
model_paths = sorted(glob.glob(os.path.join(str(get_models_dir()), "**/*.keras"), recursive=True))

# Make sure there is at least one model
if not model_paths:
    raise FileNotFoundError("No .keras model found in models directory.")

# Load the first model
trained_model = model_paths[8] # 0-2: LSTM_STFT_Dense, 3-5: LSTM_STFT, 6-8: LSTM
print(f"Converting model: {trained_model}")
model = tf.keras.models.load_model(trained_model, custom_objects={'stft_layer': stft_layer})

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# Saves in the same directory as this file
print("TFLite model saved as model.tflite")
