import pandas as pd


class CryptoDataLoader:

    _RATIOS = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']

    def __init__(self):
        self.main_df = pd.DataFrame()
        self.default_ratio = 'BTC-USD'
        self.default_future_period = 10
        self.default_seq_len = 60

    def setup_data(self):
        self._load_data()
        self._add_prediction_cols()

    def _load_data(self):
        for ratio in self._RATIOS:
            dataset = f'datasets/crypto_data/{ratio}.csv'
            df = pd.read_csv(dataset, names=['ts', 'low', 'high', 'open', 'close', 'vol'],
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


class CryptoRNN: ...
