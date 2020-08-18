import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, InputLayer, Dense
from tensorflow.keras.utils import normalize
from tensorflow.keras.datasets import mnist
from tensorflow import nn
from numpy import argmax


class MnistDeepLearningModeller:

    def __init__(self, flat_input: bool):

        x1, y1, x2, y2 = self.load_data(normalize=True)
        self.x_train = x1
        self.y_train = y1
        self.x_test = x2
        self.y_test = y2
        self.model = self.build_dl_model_and_input(flatten=flat_input)

    def build_and_validate_model(self, dense_func_sizes: tuple, epochs: int) -> tuple:
        self.add_dense_functions(dense_func_sizes)
        self.add_output_probability_softmax(10)
        self.compile_model_basic()
        self.fit_model(self.x_train, self.y_train, epochs=epochs)
        return self.validate_model(self.x_test, self.y_test)

    @classmethod
    def load_data(cls, normalize: bool =True) -> tuple:
        print('loading data...')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if normalize:
            x_train, x_test = cls.normalize_x_data(x_train, x_test)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def normalize_x_data(x_train: list, x_test: list) -> tuple:
        print('normalizing data')
        return normalize(x_train), normalize(x_test)

    def build_dl_model_and_input(self, flatten: bool) -> Model:
        print(f'building model with {"flattened" if flatten else "regular"} input layer.')
        model = Sequential()
        model.add(Flatten() if flatten else InputLayer())
        return model

    def add_dense_functions(self, sizes: tuple) -> None:
        for layer_size in sizes:
            print(f'added dense layer with size {layer_size}')
            self.model.add(Dense(layer_size, activation=nn.relu))

    def add_output_probability_softmax(self, num_outputs: int) -> None:
        print(f'added softmax final layer with size {num_outputs}')
        self.model.add(Dense(num_outputs, activation=nn.softmax))

    def compile_model_basic(self) -> None:
        print(f'compiling....')
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def fit_model(self, x: list, y: list, epochs: int =5):
        print(f'fitting model for {epochs} epochs')
        self.model.fit(x, y, epochs=epochs)

    def validate_model(self, x: list, y: list):
        return self.model.evaluate(x, y)

    def predict(self, test_list: list =None, index: int =None, raw: bool =False):

        result = self.model.predict([test_list or self.x_test])
        if index:
            result = result[index]
        if raw:
            result = argmax(result) if index else [argmax(x) for x in result]
        return result


if __name__ == '__main__':
    modeller = MnistDeepLearningModeller(flat_input=True)
    modeller.build_and_validate_model(dense_func_sizes=(128, 128),
                                      epochs=4)
    x = modeller.predict(index=5, raw=True)
    print(x)
