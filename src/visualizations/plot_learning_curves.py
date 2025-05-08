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

    # Get final values
    final_loss = history['loss'][-1] if 'loss' in history and len(history['loss']) > 0 else None
    final_val_loss = history.get('val_loss', [None])[-1]
    final_metric = history.get(metric, [None])[-1]
    final_val_metric = history.get(f'val_{metric}', [None])[-1]

    # Add text box with final values
    textstr = (
        f'Final Loss: {final_loss:.4f}\n'
        f'Final Val Loss: {final_val_loss:.4f}\n'
        f'Final {metric.capitalize()}: {final_metric:.4f}\n'
        f'Final Val {metric.capitalize()}: {final_val_metric:.4f}'
    )
    plt.gcf().text(0.5, -0.05, textstr, fontsize=12, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.7))

    # plt.tight_layout(rect=[0, 0.05, 1, 1])

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
    bars = axes[1].bar(class_names, f1s, color='skyblue')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Per-Class F1 Score')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('F1 Score')

    # Add F1 score values above bars
    for bar, f1 in zip(bars, f1s):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, height + 0.03, f'{f1:.2f}',
                     ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()