import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

from src.data.data_helper import get_raw_data_as_dataframe, segment_data
from src.models.model_components.preprocessor import SignalPreprocessor
from src.models.LSTM.LSTM import LSTM
from src.models.LSTM_STFT.LSTM_STFT import LSTM_STFT
from src.models.LSTM_STFT_Dense.LSTM_STFT_Dense import LSTM_STFT_Dense
from src.utils.path_utils import get_models_dir

PREPROC_VARIANTS = (1, 2, 3)

ARCH_CONFIG: Dict[str, Dict] = {
    "LSTM": {
        "cls": LSTM,
        "search_folder": "LSTM_search",
        "extra_hp": []            # no STFT parameters
    },
    "LSTM_STFT": {
        "cls": LSTM_STFT,
        "search_folder": "LSTM_STFT_search",
        "extra_hp": ["stft_frame_length", "stft_frame_step"],
    },
    "LSTM_STFT_Dense": {
        "cls": LSTM_STFT_Dense,
        "search_folder": "LSTM_STFT_Dense_search",
        "extra_hp": ["stft_frame_length", "stft_frame_step"],
    }
}
EARLY_STOP = tf.keras.callbacks.EarlyStopping(
    monitor="val_f1_score", mode="max", patience=5, restore_best_weights=True
)
EPOCHS = 25
SUBJECT_IDS = [1,2,3,4,5,6,7,8,9]

def get_training_data_for_subject(
    val_subject: int, preproc_variant: int
):
    """
    Replicates the original get_training_data(...) function but allows one
    arbitrary *subject* to serve as the validation set.
    """
    raw_train, raw_val = get_raw_data_as_dataframe(validation_subjects=(val_subject,))

    preproc = SignalPreprocessor(
        pre_processor_variant=preproc_variant,
        low_freq=20.0,
        high_freq=500.0,
        fs=5000.0,
        order=7,
    )

    win_len, overlap = 200 * 5, 50 * 5  # identical to the originals
    seg_train = segment_data(raw_train, window_length=win_len, overlap=overlap)
    seg_val = segment_data(raw_val, window_length=win_len, overlap=overlap)

    num_classes = pd.concat([seg_train["label"], seg_val["label"]]).nunique()
    y_train = tf.keras.utils.to_categorical(seg_train["label"], num_classes)
    y_val = tf.keras.utils.to_categorical(seg_val["label"], num_classes)

    X_train = np.stack(seg_train.drop(columns=["label", "source"])["window_data"].values)
    X_val = np.stack(seg_val.drop(columns=["label", "source"])["window_data"].values)

    X_train = preproc.batch_pre_process(X_train)
    X_val = preproc.batch_pre_process(X_val)

    input_shape = X_train.shape[1]  # matches the original scripts (an int)

    return X_train, y_train, X_val, y_val, num_classes, input_shape


def best_trial_from_folder(trial_folder: Path):
    """
    Unchanged utility that walks over keras-tuner trial sub-dirs and
    returns (best_val_f1, best_hyperparameters_dict).
    """
    best_val_f1, best_hp, best_trial = -np.inf, None, None

    for trial_subdir in trial_folder.glob("trial_*"):
        # try trial.json or trial_summary.json
        json_path = None
        for name in ("trial.json", "trial_summary.json"):
            p = trial_subdir / name
            if p.exists():
                json_path = p
                break
        if not json_path:
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        # locate metric – identical to your code
        metric_val = None
        if "metrics" in data:
            m = data["metrics"]["metrics"].get("val_f1_score", None)
            if isinstance(m, dict) and "best" in m:
                metric_val = m["best"]
            elif isinstance(m, list) and len(m):
                metric_val = max(v["value"][0] for v in m if "value" in v)
        if metric_val is None:
            metric_val = data.get("best_val_f1_score", data.get("score"))
        if metric_val is None:
            continue

        if metric_val > best_val_f1:
            best_val_f1, best_trial = metric_val, trial_subdir
            hp_container = data.get("hyperparameters", data.get("values"))
            best_hp = (
                hp_container["values"] if isinstance(hp_container, dict) and
                "values" in hp_container else hp_container
            )

    return best_val_f1, best_hp, (best_trial.name if best_trial else None)


def f1_macro(y_true_one_hot, y_pred_proba) -> float:
    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return f1_score(y_true, y_pred, average="macro")


def build_model(arch_key: str, hp: Dict, input_shape, num_classes):
    """
    Create *compiled* Keras model for the given architecture using its best
    hyper-parameters dictionary.
    """
    kwargs_base = dict(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=hp["learning_rate"],
        optimizer=hp["optimizer"],
        normalization=hp["normalization"],
        dropout=hp["dropout"],
        recurrent_dropout=hp["recurrent_dropout"],
        act_dense=hp["act_dense"],
        act_lstm=hp["act_lstm"],
    )
    # add STFT parameters for the two architectures that need them
    if arch_key != "LSTM":
        kwargs_base["stft_frame_length"] = hp["stft_frame_length"]
        kwargs_base["stft_frame_step"] = hp["stft_frame_step"]

    model_cls = ARCH_CONFIG[arch_key]["cls"]
    return model_cls(**kwargs_base).get_model()

def run_per_subject_cv():
    subjects = SUBJECT_IDS
    all_results = []

    for arch_key, cfg in ARCH_CONFIG.items():
        print(f"\n\n########  {arch_key}  ########")
        arch_root = get_models_dir() / cfg["search_folder"]

        for pv in PREPROC_VARIANTS:
            variant_folder = arch_root / f"pre_processor_variant_{pv}"

            best_val_f1, best_hp, best_trial = best_trial_from_folder(variant_folder)
            if best_hp is None:
                print(f"  • variant {pv}:  NO hyper-parameter file found – skipped ")
                continue

            print(
                f"  • variant {pv}:  best val_f1={best_val_f1:.4f}  "
                f"(trial {best_trial})"
            )

            for s in subjects:
                (
                    x_tr,
                    y_tr,
                    x_val,
                    y_val,
                    n_classes,
                    in_shape,
                ) = get_training_data_for_subject(s, pv)

                model = build_model(arch_key, best_hp, in_shape, n_classes)

                model.fit(
                    x_tr,
                    y_tr,
                    validation_data=(x_val, y_val),
                    batch_size=int(best_hp["batch_size"]),
                    epochs=EPOCHS,
                    callbacks=[EARLY_STOP],
                    verbose=0,
                )

                # evaluate on the left-out subject
                y_pred = model.predict(x_val, verbose=0)
                fold_f1 = f1_macro(y_val, y_pred)

                all_results.append(
                    dict(
                        architecture=arch_key,
                        preproc_variant=pv,
                        val_subject=s,
                        f1=fold_f1,
                        hp_trial=best_trial,
                    )
                )
                print(
                    f"      subject {s:>2}:  fold-F1 = {fold_f1:.4f}  "
                    f"(train {len(x_tr)}, val {len(x_val)})"
                )

        # after each architecture: write its results so far
        df_arch = pd.DataFrame([r for r in all_results if r["architecture"] == arch_key])
        if not df_arch.empty:
            out_file = arch_root / "cv_results.csv"
            df_arch.to_csv(out_file, index=False)
            print(f"--> interim results written to {out_file}")

    # final joined csv (all architectures)
    df_all = pd.DataFrame(all_results)
    final_out = get_models_dir() / "per_subject_cv_results.csv"
    df_all.to_csv(final_out, index=False)
    print(f"\nALL DONE – aggregated results written to {final_out}")

if __name__ == "__main__":
    run_per_subject_cv()