import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay

def plot_learning_curves(model, metric='f1_score', plot_title='Learning Curves'):
    """
    Plots the learning curves for loss and a given metric from a model's training history.

    Args:
        model: Trained model with a `history` attribute containing training history.
        metric: The metric to plot (default: 'F1 Score').
    """
    history = model.history.history if hasattr(model, 'history') else model.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(plot_title, fontsize=16)

    # Plot Loss
    axes[0].plot(history['loss'], label='Train Loss')
    axes[0].plot(history.get('val_loss', []), label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot Metric
    axes[1].plot(history.get(metric, []), label=f'Train {metric.capitalize()}')
    axes[1].plot(history.get(f'val_{metric}', []), label=f'Val {metric.capitalize()}')
    axes[1].set_title(f'{metric.capitalize()} Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(metric.capitalize())
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_and_f1(model, X_val, y_val, class_names=["rest", "grip", "hold", "release"], plot_title='Learning Curves'):
    """
    Plots the confusion matrix and per-class F1 scores side by side.

    Args:
        model: Trained Keras model or wrapper with .predict().
        X_val: Validation features.
        y_val: Validation labels (one-hot or integer).
        class_names: Optional list of class names.
    """
    # Convert one-hot to class indices if needed
    if y_val.ndim > 1 and y_val.shape[1] > 1:
        y_true = np.argmax(y_val, axis=1)
    else:
        y_true = y_val

    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    f1s = f1_score(y_true, y_pred, average=None)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(plot_title, fontsize=16)

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Confusion Matrix')

    # F1 score bar plot
    axes[1].bar(class_names, f1s, color='skyblue')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Per-Class F1 Score')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('F1 Score')

    plt.tight_layout()
    plt.show()