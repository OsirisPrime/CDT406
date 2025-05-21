import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from src.data.data_helper import get_raw_data_as_dataframe, segment_data
from src.models.model_components.preprocessor import SignalPreprocessor
from src.models.LSTM_STFT.LSTM_STFT import LSTM_STFT
from src.utils.path_utils import get_models_dir

# -------------------------- Data loading & model_components --------------------------

def get_training_data(pre_processor_variant = 1):
    # 1) load raw train/val split
    raw_train, raw_val = get_raw_data_as_dataframe(validation_subjects=(1, 2))

    # 2) build and calibrate filter
    pre_processor = SignalPreprocessor(pre_processor_variant = pre_processor_variant,
                                       low_freq=20.0,
                                       high_freq=500.0,
                                       fs=5000.0,
                                       order=7)

    # 3) segment into windows
    window_length = 200 * 5   # 200 seconds × 5 kHz = samples
    overlap       = 50  * 5
    seg_train = segment_data(raw_train, window_length=window_length, overlap=overlap)
    seg_val   = segment_data(raw_val, window_length=window_length, overlap=overlap)

    # 4) one-hot encode labels
    all_labels = pd.concat([seg_train['label'], seg_val['label']])
    num_classes = all_labels.nunique()
    y_train = tf.keras.utils.to_categorical(seg_train['label'], num_classes)
    y_val   = tf.keras.utils.to_categorical(seg_val['label'],   num_classes)

    # 5) stack windows, apply model_components
    X_train = np.stack(seg_train.drop(columns=['label','source'])['window_data'].values)
    X_val   = np.stack(seg_val.drop(columns=['label','source'])['window_data'].values)
    X_train = pre_processor.batch_pre_process(X_train)
    X_val   = pre_processor.batch_pre_process(X_val)

    input_shape = X_train.shape[1]
    return X_train, y_train, X_val, y_val, num_classes, input_shape

# -------------------------- HyperModel definition --------------------------

class LSTM_STFTHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        lr    = hp.Float('learning_rate', 1e-6, 5e-3, sampling='log')
        opt   = hp.Choice('optimizer', ['adam', 'rmsprop', 'nadam'])
        norm  = hp.Choice('normalization', ['none', 'batch', 'layer'])
        drop  = hp.Float('dropout', 0.0, 0.5, step=0.1)
        rdrop = hp.Float('recurrent_dropout', 0.0, 0.5, step=0.1)
        stft_fl = hp.Int('stft_frame_length', 32, 256)
        stft_fs = hp.Int('stft_frame_step', 8, 31)
        hp.Choice('batch_size', [32, 64, 128, 256, 512])

        model = LSTM_STFT(input_shape=self.input_shape,
                     num_classes=self.num_classes,
                     learning_rate=lr,
                     optimizer=opt,
                     normalization=norm,
                     dropout=drop,
                     recurrent_dropout=rdrop,
                     stft_frame_length=stft_fl,
                     stft_frame_step=stft_fs).get_model()
        return model

    def fit(self, hp, model, X, y, validation_data, **kwargs):
        batch_size = hp.get('batch_size')
        return model.fit(
            X, y,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=kwargs.get('epochs', 10),
            verbose=2
        )

if __name__ == "__main__":
    # Define the variants to test
    pre_processor_variants = [1, 2, 3]

    # Initialize lists to store results
    best_f1_scores = []
    best_hps = []

    for pre_processor_variant in pre_processor_variants:
        print(f"--- Testing pre_processor_variant = {pre_processor_variant} ---")

        # 1) Prepare data
        print("--- Loading and model_components data ---")
        X_train, y_train, X_val, y_val, num_classes, input_shape = get_training_data(pre_processor_variant=pre_processor_variant)
        print("--- Data loaded ---")

        # 2) Early stopping on validation F1
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            patience=5,
            restore_best_weights=True
        )

        # 3) Build the tuner
        print("--- Building the hypermodel ---")
        hypermodel = LSTM_STFTHyperModel(input_shape, num_classes)
        model_dir  = get_models_dir() / "LSTM_STFT_search"
        tuner = kt.BayesianOptimization(
            hypermodel,
            objective=kt.Objective("val_f1_score", direction="max"),
            max_trials=15,
            directory=str(model_dir),
            project_name=f"pre_processor_variant_{pre_processor_variant}",
            overwrite=True
        )
        print("--- Hypermodel built ---")

        # 4) Run hyperparameter search
        print("--- Starting hyperparameter search ---")
        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            # callbacks=[stop_early],
            verbose=2
        )

        print("--- Hyperparameter search complete ---")

        # 5) Fetch the best result
        best_hp    = tuner.get_best_hyperparameters(1)[0]
        best_trial = tuner.oracle.get_best_trials(1)[0]
        best_f1    = best_trial.metrics.get_best_value('val_f1_score')

        best_f1_scores.append(best_f1)
        best_hps.append(best_hp)

    # Print the best results for all variants
    for i, pre_processor_variant in enumerate(pre_processor_variants):
        best_f1 = best_f1_scores[i]
        best_hp = best_hps[i]

        print(f"\n--- Results for pre_processor_variant = {pre_processor_variant} ---")
        print(f"Best val_f1_score       = {best_f1:.4f}")
        print(f"learning_rate           = {best_hp.get('learning_rate')}")
        print(f"optimizer               = {best_hp.get('optimizer')}")
        print(f"normalization           = {best_hp.get('normalization')}")
        print(f"batch_size              = {best_hp.get('batch_size')}")
        print(f"dropout                 = {best_hp.get('dropout')}")
        print(f"recurrent_dropout       = {best_hp.get('recurrent_dropout')}")
        print(f"stft_frame_length       = {best_hp.get('stft_frame_length')}")
        print(f"stft_frame_step         = {best_hp.get('stft_frame_step')}")