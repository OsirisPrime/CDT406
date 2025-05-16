import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

from src.data.data_helper import get_raw_data_as_dataframe, segement_data
from src.models.model_components.preprocessor import SignalPreprocessor
from src.models.LSTM_STFT.LSTM_STFT import LSTM_STFT
from src.utils.path_utils import get_models_dir

# -------------------------- Data loading & model_components --------------------------

def get_training_data(pre_processor_variant=1):
    raw_train, raw_val = get_raw_data_as_dataframe(validation_subjects=(1, 2))

    pre_processor = SignalPreprocessor(pre_processor_variant=pre_processor_variant,
                                       low_freq=20.0,
                                       high_freq=500.0,
                                       fs=5000.0,
                                       order=7)
    pre_processor.calibrate(raw_train)

    window_length = 200 * 5  # 200 ms × 5 kHz
    overlap = 50 * 5
    seg_train = segement_data(raw_train, window_length=window_length, overlap=overlap)
    seg_val = segement_data(raw_val, window_length=window_length, overlap=overlap)

    all_labels = pd.concat([seg_train['label'], seg_val['label']])
    num_classes = all_labels.nunique()
    y_train = tf.keras.utils.to_categorical(seg_train['label'], num_classes)
    y_val = tf.keras.utils.to_categorical(seg_val['label'], num_classes)

    X_train = np.stack(seg_train.drop(columns=['label', 'source'])['window_data'].values)
    X_val = np.stack(seg_val.drop(columns=['label', 'source'])['window_data'].values)
    X_train = pre_processor.batch_pre_process(X_train)
    X_val = pre_processor.batch_pre_process(X_val)

    input_shape = X_train.shape[1]
    return X_train, y_train, X_val, y_val, num_classes, input_shape

# -------------------------- Trial parsing --------------------------

def get_best_trial_info(trial_folder: Path):
    best_val_f1 = -float('inf')
    best_hp = None
    best_trial_id = None

    for trial_subdir in trial_folder.glob("trial_*"):
        trial_json_path = None
        for candidate_name in ["trial.json", "trial_summary.json"]:
            candidate_path = trial_subdir / candidate_name
            if candidate_path.exists():
                trial_json_path = candidate_path
                break
        if trial_json_path is None:
            continue

        with open(trial_json_path, 'r') as f:
            trial_data = json.load(f)

        metric = None
        if "metrics" in trial_data:
            metrics_dict = trial_data.get("metrics", {})
            if "metrics" in metrics_dict and "val_f1_score" in metrics_dict["metrics"]:
                vals = metrics_dict["metrics"]["val_f1_score"]
                if isinstance(vals, dict) and "best" in vals:
                    metric = vals["best"]
                elif isinstance(vals, list) and len(vals) > 0:
                    metric = max(v["value"][0] for v in vals if "value" in v)

        if metric is None:
            metric = trial_data.get("best_val_f1_score", trial_data.get("score", None))
        if metric is None:
            continue

        if metric > best_val_f1:
            best_val_f1 = metric
            best_trial_id = trial_subdir.name

            hp = trial_data.get("hyperparameters", trial_data.get("values", {}))
            best_hp = hp["values"] if isinstance(hp, dict) and "values" in hp else hp

    return best_val_f1, best_hp, best_trial_id


# -------------------------- Training --------------------------

def build_and_train_best_model(input_shape, num_classes, best_hp, X_train, y_train, X_val, y_val):
    model = LSTM_STFT(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=best_hp['learning_rate'],
        optimizer=best_hp['optimizer'],
        normalization=best_hp['normalization'],
        dropout=best_hp['dropout'],
        recurrent_dropout=best_hp['recurrent_dropout'],
        act_dense=best_hp['act_dense'],
        act_lstm=best_hp['act_lstm'],
        stft_frame_length=best_hp['stft_frame_length'],
        stft_frame_step=best_hp['stft_frame_step']
    ).get_model()

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_score',
        mode='max',
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=int(best_hp['batch_size']),
        epochs=25,
        callbacks=[stop_early],
        verbose=2
    )

    return model

# -------------------------- Saving --------------------------

def save_best_model(model, pre_processor_variant):
    model_dir = get_models_dir() / "LSTM_STFT_search" / "best_LSTM_STFT_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"LSTM_STFT_variant_{pre_processor_variant}.keras")

def get_results_and_save_models(folder_path, variant_folders):
    results = []
    confusion_matrices = []

    for variant_num, variant_folder in variant_folders.items():
        print(f"\n=== Preprocessor variant {variant_num} ===")

        trial_dir = Path(folder_path) / variant_folder
        best_val_f1, best_hp, best_trial_id = get_best_trial_info(trial_dir)

        print(f"Best val_f1_score from tuning: {best_val_f1:.4f}")
        print(f"Best trial folder: {best_trial_id}")
        print(f"[INFO] Best hyperparameters:\n{json.dumps(best_hp, indent=2)}")

        if best_hp is None:
            print("No hyperparameter data found.")
            continue

        X_train, y_train, X_val, y_val, num_classes, input_shape = get_training_data(variant_num)

        model = build_and_train_best_model(
            input_shape=input_shape,
            num_classes=num_classes,
            best_hp=best_hp,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        save_best_model(model, variant_num)

        # --- Predictions & Metrics ---
        y_val_pred_prob = model.predict(X_val)
        y_val_pred = np.argmax(y_val_pred_prob, axis=1)
        y_val_true = np.argmax(y_val, axis=1)

        acc = accuracy_score(y_val_true, y_val_pred)
        val_f1 = f1_score(y_val_true, y_val_pred, average='macro')

        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Validation F1-score: {val_f1:.4f}")

        # --- Confusion matrix ---
        cm = confusion_matrix(y_val_true, y_val_pred)
        confusion_matrices.append((variant_num, cm, acc))

        results.append({
            "pre_processor_variant": variant_num,
            "best_val_f1_score": val_f1,
            "validation_accuracy": acc,
            "best_trial_id": best_trial_id,
            "hyperparameters": best_hp
        })

    # --- Plot confusion matrices ---
    num_variants = len(confusion_matrices)
    fig, axs = plt.subplots(1, num_variants, figsize=(6 * num_variants, 5))

    if num_variants == 1:
        axs = [axs]

    class_names = ["Rest", "Grip", "Hold", "Release"]

    for ax, (variant_num, cm, acc) in zip(axs, confusion_matrices):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', ax=ax, colorbar=False)
        ax.set_title(f'Variant {variant_num}\nAcc: {acc:.3f}')

    plt.suptitle("Confusion Matrices for All Variants (LSTM_STFT)", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    conf_matrix_path = get_models_dir() / "LSTM_STFT_search" / "best_LSTM_STFT_models" / "confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    plt.show()

    return results

# -------------------------- Entry point --------------------------

if __name__ == "__main__":
    LSTM_search_folder_path = str(get_models_dir()) + "/LSTM_STFT_search"

    variant_folders = {
        1: "pre_processor_variant_1",
        2: "pre_processor_variant_2",
        3: "pre_processor_variant_3",
    }

    LSTM_results = get_results_and_save_models(LSTM_search_folder_path, variant_folders)

    print("\nModel training and saving complete.")
