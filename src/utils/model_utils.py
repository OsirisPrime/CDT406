import tensorflow as tf
from src.utils.path_utils import get_models_dir

def save_best_model(model, model_name):
    """Save model if it's the best so far using TensorFlow."""
    model_dir = get_models_dir() / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{model_name}.h5"
    model_path = model_dir / file_name
    model.save(model_path)
    return model_path

def load_model(model, model_name):
    """Load saved model weights into an existing TensorFlow model."""
    model_dir = get_models_dir() / model_name
    file_name = f"{model_name}.h5"
    model_path = model_dir / file_name
    model.load_weights(model_path)
    return model