from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam

class RNNMnistModel:

    def __init__(self):
        self.model = None
        self.loaded = False


    def build_model(self, lstm_sizes: tuple, dropout_size: float):
        self.model = Sequential()
        if not self.loaded:
            self.load_mnist_dataset()
        self.model.add(LSTM(lstm_sizes[0], input_shape=(self.x_train.shape[1:]), return_sequences=True))
        self.model.add(Dropout(dropout_size))
        for size in lstm_sizes[1:]:
            self.model.add(LSTM(size)) # Not using relu as leaving tanh default allows CUDNN fast training
            self.model.add(Dropout(dropout_size))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(dropout_size))
        self.model.add(Dense(10, activation='softmax'))

    def compile_and_validate_model(self, learning_rate:float, decay:float, epochs: int):
        if self.model:
            self.model.compile(loss='sparse_categorical_crossentropy',
                               optimizer=Adam(lr=learning_rate, decay=decay),
                               metrics=['accuracy'])
            self.model.fit(self.x_train, self.y_train, epochs=epochs, validation_data=(self.x_test, self.y_test))

    def load_mnist_dataset(self):
        (x_train, self.y_train), (x_test, self.y_test) = load_data()
        self.x_train = x_train / 255.0
        self.x_test = x_test / 255.0
        self.loaded = True

if __name__ == '__main__':
    modl = RNNMnistModel()
    modl.build_model(lstm_sizes=(128, 128),
                     dropout_size=.2)
    modl.compile_and_validate_model(1e-3, 1e-5, 3)