import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import deque
from numpy import array, asarray
from pandas import DataFrame, read_csv
from random import shuffle
from sklearn.preprocessing import scale
from pickle import load, dump
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.models import load_model
from datetime import datetime as dt


class CryptoDataLoader:

    _RATIOS = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']
    default_ratio = 'BTC-USD'
    default_seq_len = 60
    default_future_period = 10

    def __init__(self, seq_len: int =None):
        self.main_df = DataFrame()
        self.seq_len = seq_len or self.default_seq_len

    def setup_training_data(self, percentage_separation: float, load_saved: bool =True, save: bool =True,
                            ratio: str =None, future_period: int =None):
        if load_saved:
            try:
                return self._load_saved_training_data()
            except: ...
        self._load_csv_data()
        self._add_prediction_cols(ratio or self.default_ratio, future_period=future_period or self.default_future_period)
        x1, y1, x2, y2 = self._separate_training_data(pct=percentage_separation)
        if save:
            try:
                self._save_training_data(x1, y1, x2, y2)
            except: ...
        return x1, y1, x2, y2

    def _load_csv_data(self):
        for ratio in self._RATIOS:
            dataset = f'datasets/crypto_data/{ratio}.csv'
            df = read_csv(dataset, names=['ts', 'low', 'high', 'open', 'close', 'vol'])
            df.rename(columns={'close': f'{ratio}_close', 'vol': f'{ratio}_vol'}, inplace=True)
            df.set_index("ts", inplace=True)
            df = df[[f'{ratio}_close', f'{ratio}_vol']]

            self.main_df = df if len(self.main_df) == 0 else self.main_df.join(df)

        self.main_df.fillna(method="ffill", inplace=True)
        self.main_df.dropna(inplace=True)

    @staticmethod
    def classify(curr_price: float, future_price: float):
        return 1 if float(future_price) > float(curr_price) else 0

    def _add_prediction_cols(self, ratio, future_period):
        self.main_df['future'] = self.main_df[f'{ratio}_close'].shift(-future_period)
        self.main_df['target'] = list(map(self.classify,
                                          self.main_df[f'{ratio}_close'],
                                          self.main_df['future']))
        self.main_df.dropna(inplace=True)

    def _separate_training_data(self, pct: float =0.1):
        times = sorted(self.main_df.index.values)
        last_x_pct_ts = sorted(self.main_df.index.values)[-int(pct * len(times))]

        validation_main_df = self.main_df[(self.main_df.index >= last_x_pct_ts)]
        main_df = self.main_df[(self.main_df.index < last_x_pct_ts)]

        train_x, train_y = self._preprocess_data(main_df)
        test_x, test_y = self._preprocess_data(validation_main_df)

        return train_x, train_y, test_x, test_y

    def _preprocess_data(self, df: DataFrame):

        df = df.drop("future", 1)

        for col in df.columns:
            if col != "target":
                df[col] = df[col].pct_change()
                df.dropna(inplace=True)
        df.dropna(inplace=True)

        sequential_data = []
        prev_mins = deque(maxlen=self.seq_len)

        for row in df.values:
            prev_mins.append([val for val in row[:-1]])
            if len(prev_mins) == self.seq_len:
                sequential_data.append([scale(array(prev_mins), copy=False), row[-1]]) #TODO: Scale above?
        shuffle(sequential_data)
        sequential_data = self._balance_data(sequential_data)
        X = []
        y = []

        for seq, target in sequential_data:
            X.append(seq)
            y.append(target)

        return array(X), array(y)

    @staticmethod
    def _balance_data(data: list):
        buys = []
        sells = []

        for seq, target in data:
            sells.append([seq, target]) if target == 0 else buys.append([seq, target])

        lower = min(len(buys), len(sells))
        data = buys[:lower] + sells[:lower]
        shuffle(data)
        return data

    def _save_training_data(self, x1, y1, x2, y2):
        with open('pickles/crypto_train_x.pickle', 'wb+') as p:
            dump(x1, p)
        with open('pickles/crypto_train_y.pickle', 'wb+') as p:
            dump(y1, p)
        with open('pickles/crypto_test_x.pickle', 'wb+') as p:
            dump(x2, p)
        with open('pickles/crypto_test_y.pickle', 'wb+') as p:
            dump(y2, p)

    def _load_saved_training_data(self):
        with open('pickles/crypto_train_x.pickle', 'rb') as p:
            x1 = load(p)
        with open('pickles/crypto_train_y.pickle', 'rb') as p:
            y1 = load(p)
        with open('pickles/crypto_test_x.pickle', 'rb') as p:
            x2 = load(p)
        with open('pickles/crypto_test_y.pickle', 'rb') as p:
            y2 = load(p)
        return x1, y1, x2, y2


class CryptoRNN:

    _LOGDIR = 'logs/crypto/'
    _SAVED_MODELS_DIR = 'models/'

    def __init__(self, tensor_board: bool =True, model_checkpt: bool =True, log_name: str =None, savename: str =None):
        self.model = None if not savename else load_model(self._SAVED_MODELS_DIR + savename)
        self.training_data = []
        self.callbacks = []
        if tensor_board:
            __log_name = log_name or f'rnn-crypto-unnamed-{dt.now().strftime("%Y%m%d-%H%M%S")}'
            __tensorboard = TensorBoard(log_dir=self._LOGDIR + __log_name)
            self.callbacks.append(__tensorboard)
        if model_checkpt:
            __cp_filepath = "RNN_Final-{epoch:02d}-{val_loss:.2f}"
            __checkpoint = ModelCheckpoint("models/{}.model".format(__cp_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
            self.callbacks.append(__checkpoint)


    def build_and_validate_model(self, lstm_layer_sizes: tuple, dense_layer_size: int, drouput_sizes: tuple,
                                 train_x, train_y, test_x, test_y, batch_size: int, epochs: int,
                                 learn_rate: float =1e-3, decay: float =1e-6):
        self.model = Sequential()
        for idx, layer_size in enumerate(lstm_layer_sizes):
            if idx == len(lstm_layer_sizes) - 1:
                self.model.add(LSTM(layer_size))
                self.model.add(Dropout(drouput_sizes[idx]))
                self.model.add(BatchNormalization())
                self.model.add(Dense(dense_layer_size, activation='relu'))
                self.model.add(Dropout(drouput_sizes[idx + 1]))
            else:
                if idx == 0:
                    self.model.add(LSTM(layer_size, input_shape=train_x.shape[1:], return_sequences=True))
                else:
                    self.model.add(LSTM(layer_size, return_sequences=True))
                self.model.add(Dropout(drouput_sizes[idx]))
                self.model.add(BatchNormalization())

        self.model.add(Dense(2, 'softmax'))
        opt = Adam(lr=learn_rate, decay=decay)
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(train_x, train_y,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(test_x, test_y),
                       callbacks=[])
