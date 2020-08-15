import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from random import shuffle


class CatsVsDogsModeller:

    _DATADIR = '../kaggle'
    _CATEGORIES = {'Cat': 0,
                   'Dog': 1}

    def __init__(self):
        self.training_data = []

    def build_and_validate_model(self, dense_func_sizes: tuple, epochs: int) -> tuple:
        ...

    def load_data(self, img_size_x, img_size_y=None, load_saved=True):
        if load_saved:
            try:
                return self.load_training_data()
            except: ...
        for category in self._CATEGORIES.keys():
            path = os.path.join(self._DATADIR, category)
            self.build_dataset(path, category, img_size_x, img_size_y)
        shuffle(self.training_data)
        X, y = self.prepare_training_data(img_size_x, img_size_y)
        self.save_training_data(X, y)
        self.X = X
        self.y = y

    def build_dataset(self, path, category, size_x, size_y):
        category = self._CATEGORIES.get(category)
        for img in os.listdir(path):
            try:
                full_path = os.path.join(path, img)
                image_arr = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                new_arr = self._resize_image(image_arr, size_x, size_y)
                self.training_data.append((new_arr, category))
            except: ...

    @staticmethod
    def _resize_image(img: list, x: int, y:int =None):
        return cv2.resize(img, (x, y or x))

    def prepare_training_data(self, img_size_x, img_size_y=None) -> tuple:
        X = [feature for feature, _ in self.training_data]
        X = np.array(X).reshape(-1, img_size_x, img_size_y or img_size_x, 1)
        y = [classification for _, classification in self.training_data]
        return X, y

    @staticmethod
    def save_training_data(X, y):
        from pickle import dump
        with open('X.pickle', 'wb+') as x_pick:
            dump(X, x_pick)
        with open('y.pickle', 'wb+') as y_pick:
            dump(y, y_pick)

    @staticmethod
    def load_training_data():
        from pickle import load
        with open('X.pickle', 'rb') as x_pick:
            X = load(x_pick)
        with open('y.pickle', 'rb') as y_pick:
            y = load(y_pick)
        return X, y


if __name__ == '__main__':
    import time
    modeller = CatsVsDogsModeller()
    t1 = time.time()
    modeller.load_data(90)
    print(time.time() - t1)
    t2 = time.time()
    modeller.load_data(90)
    print(time.time() - t2)
