from collections import deque
from numpy import array
from pandas import DataFrame, read_csv
from random import shuffle
from sklearn.preprocessing import scale


class CryptoDataLoader:

    _RATIOS = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']
    default_ratio = 'BTC-USD'
    default_seq_len = 60
    default_future_period = 10

    def __init__(self):
        self.main_df = DataFrame()

    def setup_data(self):
        self._load_data()
        self._add_prediction_cols()
        self._separate_training_data(0.05)

    def _load_data(self):
        for ratio in self._RATIOS:
            dataset = f'datasets/crypto_data/{ratio}.csv'
            df = read_csv(dataset, names=['ts', 'low', 'high', 'open', 'close', 'vol'],
                             index_col='ts')
            df.rename(columns={'close': f'{ratio}_close',
                               'vol': f'{ratio}_vol'},
                      inplace=True)
            df = df[[f'{ratio}_close', f'{ratio}_vol']]

            self.main_df = df if len(self.main_df) == 0 else self.main_df.join(df)

    def classify(self, curr_price: float, future_price: float):
        return 1 if future_price > curr_price else 0

    def _add_prediction_cols(self, future_period:int =None):
        for ratio in self._RATIOS:
            self.main_df[f'{ratio}_future'] = self.main_df[f'{ratio}_close'].shift(-(future_period or self.default_future_period))
            self.main_df[f'{ratio}_target'] = list(map(self.classify,
                                                       self.main_df[f'{ratio}_close'],
                                                       self.main_df[f'{ratio}_future']))

    def _separate_training_data(self, pct: float =0.1):
        times = sorted(self.main_df.index.values)
        last_x_pct_ts = times[-int(pct * len(times))]
        self.validation_main_df = self.main_df[(self.main_df.index >= last_x_pct_ts)]
        self.main_df = self.main_df[(self.main_df.index < last_x_pct_ts)]
        #self._preprocess_data(self.main_df)

    @classmethod
    def _preprocess_data(cls, df: DataFrame, seq_len: int =None):
        seq_len = seq_len or cls.default_seq_len
        for col in df.columns:
            if 'future' in col:
                df = df.drop(col, 1)
            else:
                if 'target' not in col:
                    df[col] = df[col].pct_change()
                    df.dropna(inplace=True)

        sequential_data = []
        prev_mins = deque(maxlen=seq_len)

        for row in df.to_numpy():
            prev_mins.append([val for val in row[:-1]])
            if len(prev_mins) == seq_len:
                sequential_data.append([scale(array(prev_mins), copy=False), row[-1]])

        shuffle(sequential_data)


class CryptoRNN: ...
