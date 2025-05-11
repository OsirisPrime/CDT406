import numpy as np
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.callbacks import ModelCheckpoint

# Pre-processing functions
from src.data.data_helper import get_raw_data_as_dataframe
from src.models.preprocessing.preprocessor import SignalPreprocessor
from src.data.data_helper import segement_data

# Model functions
from src.models.LSTM.LSTM import LSTM
#from src.models.LSTM_STFT.LSTM_STFT import LSTM_STFT
#from src.models.LSTM_STFT_Dense.LSTM_STFT_Dense import LSTM_STFT_Dense

"""
    This code creates the 3 pre-processing and 3 models to train.
    In total, there are 9 models that will be trained. Each
    model will be trained on a different dataset.
"""

# ------------ Hyperparameters ------------ #
bandpass_order = 7
high_freq = 500.0
low_freq = 20.0
fs = 5000.0

random_val = True   # If True will randomly split the data into training and validation sets.

param_dist = {
    "model_learning_rate": [1e-2, 1e-3, 1e-4],
    "model_batch_size": [32, 64, 128],
    "model_dropout_rate": [0.1, 0.2, 0.3],             # Only if the LSTM supports it
    "model_optimizer": ["adam", "nadam", "rmsprop"]    # Only if the LSTM supports it
}

# Load and process data
raw_data = get_raw_data_as_dataframe()
pre_processor = SignalPreprocessor(low_freq=low_freq, high_freq=high_freq, fs=fs, order=bandpass_order)
pre_processor.calibrate(raw_data)

segmented_data = segement_data(raw_data, window_length=200*5, overlap=50*5)
num_classes = segmented_data['label'].nunique()

y_data = np.array(segmented_data['label'].values)
y_data = tf.keras.utils.to_categorical(y_data, num_classes=num_classes)
X_data = np.stack(segmented_data.drop(columns=['label', 'source'])['window_data'].values)
X_data = pre_processor.batch_pre_process(X_data)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2) if random_val else (
    X_data[:int(len(X_data) * 0.8)], X_data[int(len(X_data) * 0.8):],
    y_data[:int(len(y_data) * 0.8)], y_data[int(len(y_data) * 0.8):])


labels = np.argmax(y_train, axis=1)
unique, counts = np.unique(labels, return_counts=True)

# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / counts[0]) * (labels.shape[0] / 2.0)
weight_for_1 = (1 / counts[1]) * (labels.shape[0] / 2.0)
weight_for_2 = (1 / counts[2]) * (labels.shape[0] / 2.0)
weight_for_3 = (1 / counts[3]) * (labels.shape[0] / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3}


# Set up the model
LSTM_model = LSTM(
    input_shape=X_data.shape[1],
    num_classes=num_classes,
    learning_rate=1e-3
)

# Define the model
wrapped_model = KerasClassifier(
    model=LSTM_model.get_model,
    input_shape=X_data.shape[1],
    num_classes=num_classes,
    verbose=0,
    class_weight=class_weight,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5),
    ]
)

# Set up RandomSearchCV
search = RandomizedSearchCV(
    estimator=wrapped_model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='f1_macro'
)


# Perform the random search
search.fit(X_train, y_train)

# Save the best model
# search.best_estimator_.model.save("final_best_LSTM_model.h5")

