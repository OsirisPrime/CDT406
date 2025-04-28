from tensorflow.keras import layers, models

class SimpleANN:
    def __init__(self, input_shape, num_classes):
        self.model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, verbose=2):
        self.model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=verbose)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=2)