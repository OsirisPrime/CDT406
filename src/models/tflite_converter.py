import os
import glob
import tensorflow as tf
from src.utils.path_utils import get_models_dir
from stft_layer import STFTLayer

if __name__ == "__main__":
    # Get list of all .keras models in the models directory
    model_paths = sorted(glob.glob(os.path.join(str(get_models_dir()), "**/*.keras"), recursive=True))

    # Make sure there is at least one model
    if not model_paths:
        raise FileNotFoundError("No .keras model found in models directory.")

    # Load the first model
    for trained_model in model_paths:
        print(f"Converting model: {trained_model}")

        # Register the custom layer so Keras knows how to load it
        custom_objects = {'STFTLayer': STFTLayer}

        # Load the model with the custom layer
        model = tf.keras.models.load_model(trained_model, custom_objects=custom_objects)

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the converted model
        tflite_filename = os.path.splitext(os.path.basename(trained_model))[0] + ".tflite"
        tflite_path = os.path.join(os.path.dirname(trained_model), tflite_filename)

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        # Saves in the same directory as the original model
        print(f"TFLite model saved as {tflite_path}")
