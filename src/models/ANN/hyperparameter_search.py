import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from src.data.data_helper import get_raw_data_as_dataframe, segment_data
from src.models.model_components.preprocessor import SignalPreprocessor
from src.models.ANN.ANN import ANN  # Import ANN instead of LSTM
from src.utils.path_utils import get_models_dir


# -------------------------- Data loading & model_components --------------------------

def get_training_data(pre_processor_variant=1):
    # 1) load raw train/val split
    raw_train, raw_val = get_raw_data_as_dataframe(validation_subjects=(1, 2))

    # 2) build and calibrate filter
    pre_processor = SignalPreprocessor(pre_processor_variant=pre_processor_variant,
                                       low_freq=20.0,
                                       high_freq=500.0,
                                       fs=5000.0,
                                       order=7)

    # 3) segment into windows
    window_length = 200 * 5  # 200 milliseconds × 5 kHz = samples
    overlap = 50 * 5
    seg_train = segment_data(raw_train, window_length=window_length, overlap=overlap)
    seg_val = segment_data(raw_val, window_length=window_length, overlap=overlap)

    # 4) one-hot encode labels
    all_labels = pd.concat([seg_train['label'], seg_val['label']])
    num_classes = all_labels.nunique()
    y_train = tf.keras.utils.to_categorical(seg_train['label'], num_classes)
    y_val = tf.keras.utils.to_categorical(seg_val['label'], num_classes)

    # 5) stack windows, apply preprocessing
    X_train = np.stack(seg_train.drop(columns=['label', 'source'])['window_data'].values)
    X_val = np.stack(seg_val.drop(columns=['label', 'source'])['window_data'].values)
    X_train = pre_processor.batch_pre_process(X_train)
    X_val = pre_processor.batch_pre_process(X_val)

    # For ANN, we need to flatten the input shape (since we're not using time series structure)
    # If X_train is [num_samples, window_length], we keep it as is
    # If X_train is [num_samples, window_length, channels], we flatten it
    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)

    input_shape = X_train.shape[1]
    return X_train, y_train, X_val, y_val, num_classes, input_shape


# -------------------------- HyperModel definition --------------------------

class ANNHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        lr = hp.Float('learning_rate', 1e-6, 5e-3, sampling='log')
        opt = hp.Choice('optimizer', ['adam', 'rmsprop', 'nadam'])
        norm = hp.Choice('normalization', ['none', 'batch', 'layer'])
        drop = hp.Float('dropout', 0.0, 0.5, step=0.1)
        activation = hp.Choice('activation', ['tanh', 'relu', 'sigmoid'])
        hp.Choice('batch_size', [32, 64, 128, 256, 512])

        # New hyperparameters specific to the ANN architecture
        units_dense1 = hp.Int('units_dense1', 16, 128, step=16)
        units_dense2 = hp.Int('units_dense2', 8, 64, step=8)
        units_dense3 = hp.Int('units_dense3', 16, 128, step=16)

        model = ANN(input_shape=self.input_shape,
                    num_classes=self.num_classes,
                    learning_rate=lr,
                    optimizer=opt,
                    normalization=norm,
                    dropout=drop,
                    activation=activation,
                    units_dense1=units_dense1,
                    units_dense2=units_dense2,
                    units_dense3=units_dense3).get_model()
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
        print("--- Loading and preprocessing data ---")
        X_train, y_train, X_val, y_val, num_classes, input_shape = get_training_data(
            pre_processor_variant=pre_processor_variant)
        print(f"--- Data loaded with input shape: {input_shape} ---")

        # 2) Early stopping on validation F1
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            patience=5,
            restore_best_weights=True
        )

        # 3) Build the tuner
        print("--- Building the hypermodel ---")
        hypermodel = ANNHyperModel(input_shape, num_classes)
        model_dir = get_models_dir() / "ANN_search"  # Change directory name to reflect ANN model
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
            callbacks=[stop_early],
            verbose=2
        )

        print("--- Hyperparameter search complete ---")

        # 5) Fetch the best result
        best_hp = tuner.get_best_hyperparameters(1)[0]
        best_trial = tuner.oracle.get_best_trials(1)[0]
        best_f1 = best_trial.metrics.get_best_value('val_f1_score')

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
        print(f"activation              = {best_hp.get('activation')}")
        print(f"units_dense1            = {best_hp.get('units_dense1')}")
        print(f"units_dense2            = {best_hp.get('units_dense2')}")
        print(f"units_dense3            = {best_hp.get('units_dense3')}")