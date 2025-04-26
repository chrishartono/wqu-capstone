import logging
from datetime import datetime

import pandas as pd

from comovement import test_cointegration
from feature_engineering import AddFeatures
from spread import AddPolyfitSpread
from target_creation import AddPeakNeighboursSingleColumn


def SetLogging(logname: str, append: bool = False):
	mode = 'a' if append else 'w'
	logging.basicConfig(format='%(asctime)s.%(msecs)03d;%(levelname)s;{%(module)s};[%(funcName)s];%(thread)d-%(process)d;%(message)s',
						datefmt='%d/%m/%Y %I:%M:%S',
						handlers=[logging.StreamHandler(), logging.FileHandler(logname, mode=mode)],
						level=logging.INFO)

if __name__ == '__main__':
	now_str = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
	SetLogging(f'wqu_capstone_{now_str}.log', False)

	train_frac = 0.8
	last_rows = 3000
	combination = ('close_sol-usdt', 'close_avax-usdt')

	prices_df = pd.read_csv('dataset/binance_1h_ohlcv_2021-2025.csv', index_col='date', parse_dates=True)

	# test_cointegration(prices_df[list(combination)], combination)

	pairs = set([c.split('_')[1] for c in combination])
	columns_to_choose = [col for col in prices_df.columns if col != 'date' and col.split('_')[1] in pairs]

	prices_df = prices_df[columns_to_choose]
	prices_df = prices_df.iloc[-last_rows:]

	train_split_idx = int(train_frac * len(prices_df))

	train = prices_df.iloc[:train_split_idx]
	test = prices_df.iloc[train_split_idx:]

	train, coefs = AddPolyfitSpread(train, combination, coefs=None)
	test, _ = AddPolyfitSpread(test, combination, coefs)

	train_days = (train.index[-1] - train.index[0]).days
	window_days = 10
	train, test = AddFeatures(train, test, combination, window_days)
	feats_df = pd.concat([train, test], axis=0)
	window_rows = int(len(train) / 20)
	feats_df = AddPeakNeighboursSingleColumn(feats_df, target_col='spread', period=window_rows, resulting_target_column='TARGET', numNeighbours=10)
