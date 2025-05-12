import os
import glob
import tensorflow as tf
from src.utils.path_utils import get_models_dir

"""
    Converts the most recently modified trained Keras model (.h5) to a TFLite model.
    Saves it as `model.tflite` in the current working directory.
    You can then copy this file to the BeagleBone Green.
"""

# Get list of all .h5 models in the models directory
model_paths = sorted(glob.glob(os.path.join(str(get_models_dir()), "**/*.h5"), recursive=False))

# Make sure there is at least one model
if not model_paths:
    raise FileNotFoundError("No .h5 model found in models directory.")

# Load the first model
trained_model = model_paths[0]
print(f"Converting model: {trained_model}")
model = tf.keras.models.load_model(trained_model)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as model.tflite")