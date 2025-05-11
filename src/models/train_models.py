import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# Pre-processing paths
from src.data.data_helper import get_raw_data_as_dataframe
from src.models.preprocessing.preprocessor import SignalPreprocessor
from src.data.data_helper import segement_data

# Models paths
from src.models.LSTM.LSTM import LSTM
from src.models.LSTM_STFT.LSTM_STFT import LSTM_STFT
from src.models.LSTM_STFT_Dense.LSTM_STFT_Dense import LSTM_STFT_Dense

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

learning_rate = 1e-3
batch_size = 512
epochs = 20

random_val = True   # If True will randomly split the data into training and validation sets.


raw_data = get_raw_data_as_dataframe()

pre_processor = SignalPreprocessor(
    low_freq=20.0, # Maybe try down to 17.
    high_freq=500.0, # Around 100-150 looks good for our data.
    fs=5000.0,
    order=7
)

pre_processor.calibrate(raw_data)

segmented_data = segement_data(raw_data, window_length=200*5, overlap=50*5)

num_classes = segmented_data['label'].nunique()

y_data = np.array(segmented_data['label'].values)
y_data = tf.keras.utils.to_categorical(y_data, num_classes=num_classes)

X_data = np.stack(segmented_data.drop(columns=['label', 'source'])['window_data'].values)

X_data = pre_processor.batch_pre_process(X_data)


if random_val == True:
    X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data, test_size=0.2
        )
else:
    split_idx = int(len(X_data) * 0.8)  # 80% for training
    X_train = X_data[:split_idx]
    y_train = y_data[:split_idx]
    X_val = X_data[split_idx:]
    y_val = y_data[split_idx:]



label_percentages = segmented_data['label'].value_counts(normalize=True).sort_index() * 100
print(label_percentages)

# Plot label distribution
plt.figure(figsize=(8, 4))
segmented_data['label'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.show()

labels = np.argmax(y_train, axis=1)
unique, counts = np.unique(labels, return_counts=True)

# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / counts[0]) * (labels.shape[0] / 2.0)
weight_for_1 = (1 / counts[1]) * (labels.shape[0] / 2.0)
weight_for_2 = (1 / counts[2]) * (labels.shape[0] / 2.0)
weight_for_3 = (1 / counts[3]) * (labels.shape[0] / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3}


# Plot label distribution for resampled training data
plt.figure(figsize=(8, 4))
labels_resampled = np.argmax(y_train, axis=1)
unique, counts = np.unique(labels_resampled, return_counts=True)
plt.bar(unique, counts)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels (Resampled Training Data)')
plt.show()


LSTM_model = LSTM(
    input_shape=X_data.shape[1],
    num_classes=num_classes,
    learning_rate=learning_rate
)

LSTM_STFT_model = LSTM_STFT(
    input_shape=X_data.shape[1],
    num_classes=num_classes,
    learning_rate=learning_rate
)

LSTM_STFT_Dense_model = LSTM_STFT_Dense(
    input_shape=X_data.shape[1],
    num_classes=num_classes,
    learning_rate=learning_rate
)

LSTM_model.get_model().fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=2,
                       class_weight=class_weight,
                       callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                                  ModelCheckpoint(filepath='best_LSTM_model.h5',
                                                  monitor='val_loss', save_best_only=True)]
                       )

LSTM_STFT_model.get_model().fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=2,
                       class_weight=class_weight,
                       callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                                  ModelCheckpoint(filepath='best_LSTM_STFT_model.h5',
                                                  monitor='val_loss', save_best_only=True)]
                       )

LSTM_STFT_Dense_model.get_model().fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=2,
                       class_weight=class_weight,
                       callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                                  ModelCheckpoint(filepath='best_LSTM_STFT_Dense_model.h5',
                                                  monitor='val_loss', save_best_only=True)]
                       )







