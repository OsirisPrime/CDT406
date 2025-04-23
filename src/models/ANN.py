import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load EMG data
data_file = r'C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\FIN data 2\emg_data.npz'
loaded = np.load(data_file)
X = loaded['data']
y = loaded['labels']

# Encode labels if necessary
if y.dtype.type is np.str_ or y.dtype.type is np.object_:
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

# Split: 80% for training, 20% for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)

# Build the ANN model
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    verbose=1)

# Evaluate on validation set
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=2)
print(f"\n✅ Validation accuracy: {val_acc:.2f}")

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.title('Accuracy over Epochs (Train vs Validation)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
