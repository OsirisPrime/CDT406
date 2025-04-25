# BROKEN! We don't use pytorch!

import torch
from src.utils.path_utils import get_models_dir

def save_best_model(model, model_name, experiment_id=None):
    """Save model if it's the best so far."""
    model_dir = get_models_dir() / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"best_simple_cnn_{experiment_id}.pt" if experiment_id else model_dir / model_name + ".pt"
    torch.save(model.state_dict(), model_path)
    return model_path

def load_model(model, model_name, experiment_id=None):
    pass