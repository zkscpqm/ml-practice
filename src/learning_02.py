import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from random import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from datetime import datetime as dt
from tensorflow.keras.callbacks import TensorBoard


class CatsVsDogsModeller:

    _DATADIR = '../kaggle'
    _LOGDIR = '../logs/'
    _CATEGORIES = ['Cat', 'Dog']

    def __init__(self, tb: bool =True, log_name: str =None):
        self.model = Sequential()
        self.training_data = []
        self.callbacks = []
        if tb:
            _log_name = log_name or f'cats-dogs-kaggle-{dt.now().strftime("%Y%m%d-%H%M%S")}'
            _tensorboard = TensorBoard(log_dir=self._LOGDIR + _log_name)
            self.callbacks.append(_tensorboard)

    def build_and_validate_model(self, conv_layer_sizes: tuple, conv_window_size: int, dense_layers: tuple,
                                 batch_size: int, val_split: float, epochs: int):
        print(f'Building model with {conv_layer_sizes}-c-{dense_layers}-d-{conv_window_size}-w-{batch_size}-b-{val_split}')

        self.model.add(Conv2D(conv_layer_sizes[0], (conv_window_size, conv_window_size),
                              input_shape=self.X.shape[1:], activation='relu'))
        self.model.add(MaxPooling2D())
        for size in conv_layer_sizes[1:]:
            self.model.add(Conv2D(size, (conv_window_size, conv_window_size), activation='relu'))
            self.model.add(MaxPooling2D())

        print('Flattening')
        self.model.add(Flatten())
        for size in dense_layers:
            self.model.add(Dense(size, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        print('Compiling model')
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        print('Fitting model')
        self.model.fit(self.X, self.y, batch_size=batch_size, validation_split=val_split,
                       epochs=epochs, callbacks=self.callbacks)

    def load_data(self, img_size_x, img_size_y=None, load_saved=True):
        X = []
        y = []
        if load_saved:
            try:
                X, y = self.load_training_data()
            except: print('Load unsuccessful')
        if len(X) == 0:
            print('Building dataset')
            for category in self._CATEGORIES:
                path = os.path.join(self._DATADIR, category)
                self.build_dataset(path, category, img_size_x, img_size_y)
            shuffle(self.training_data)
            print('Preparing training data')
            X, y = self.prepare_training_data(img_size_x, img_size_y)
        self.X = X / 255.0 # Normalizing data
        self.y = y
        print('Saving training data')
        self.save_training_data(X, y)

    def build_dataset(self, path, category, size_x, size_y):
        category = self._CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                full_path = os.path.join(path, img)
                image_arr = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                new_arr = cv2.resize(image_arr, (size_x, size_y or size_x))
                self.training_data.append((new_arr, category))
            except: ...

    def prepare_training_data(self, img_size_x, img_size_y=None) -> tuple:
        X = []
        y = []
        for feature, classification in self.training_data:
            X.append(feature)
            y.append(classification)
        X = np.array(X).reshape(-1, img_size_x, img_size_y or img_size_x, 1)
        y = np.array(y)
        return X, y

    @staticmethod
    def save_training_data(X, y):
        from pickle import dump
        with open('X.pickle', 'wb+') as x_pick:
            dump(X, x_pick)
        with open('y.pickle', 'wb+') as y_pick:
            dump(y, y_pick)

    def load_training_data(self):
        print('Attempting to load from file')
        from pickle import load
        with open('X.pickle', 'rb') as x_pick:
            X = load(x_pick)
        with open('y.pickle', 'rb') as y_pick:
            y = load(y_pick)
        print('Load successful')
        return X, y



if __name__ == '__main__':

    dense_layers = ((32,), (64,),
                    (32, 32), (64, 64),
                    (32, 64), (64, 32))
    conv_layers = (
        (64, 64, 64),
        (128, 128, 128),
        (64, 64, 128),
        (128, 128, 256),
        (64, 64, 128),
        (128, 128, 256),
        (64, 128, 128),
        (128, 256, 256),
        (64, 128, 64),
        (128, 256, 128),
    )

    window_sizes = (3, 4)
    batch_sizes = (16, 32, 64)
    splits = (0.1, 0.2, 0.3)

    for dense_layer_tuple in dense_layers:
        for conv_layer_tuple in conv_layers:
            for window_size in window_sizes:
                for batch_size in batch_sizes:
                    for val_split in splits:
                        if val_split == .3:
                            if random.randint(0, 1) == 1:
                                continue
                        name = f'{conv_layer_tuple}-c--{dense_layer_tuple}-d--{window_size}-w--{batch_size}-b--{val_split}-v--{dt.now().strftime("%Y%m%d-%H%M%S")}-timeid'
                        modeller = CatsVsDogsModeller(log_name=name)
                        modeller.load_data(80, load_saved=True)
                        modeller.build_and_validate_model(conv_layer_sizes=conv_layer_tuple,
                                                          conv_window_size=window_size,
                                                          dense_layers=dense_layer_tuple,
                                                          batch_size=batch_size,
                                                          val_split=val_split,
                                                          epochs=10)
