import logging
from datetime import datetime

import numpy as np
import pandas as pd

from backtester import Backtester
from combinations import CreateAllPossibleCombinations
from comovement import ComovementType, test_cointegration
from feature_engineering import AddFeatures
from spread import AddPolyfitSpread
from target_creation import AddPeakNeighboursSingleColumn


def SetLogging(logname: str, append: bool = False):
	mode = 'a' if append else 'w'
	logging.basicConfig(format='%(asctime)s.%(msecs)03d;%(levelname)s;{%(module)s};[%(funcName)s];%(thread)d-%(process)d;%(message)s',
						datefmt='%d/%m/%Y %I:%M:%S',
						handlers=[logging.StreamHandler(), logging.FileHandler(logname, mode=mode)],
						level=logging.INFO)

def manual_test(prices_df: pd.DataFrame):
	train_frac = 0.8
	last_rows = 3000
	combination = ('close_sol-usdt', 'close_avax-usdt')

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
	# feats_df = pd.concat([train, test], axis=0)
	# window_rows = int(len(train) / 20)
	# feats_df = AddPeakNeighboursSingleColumn(feats_df, combination, target_col='spread', period=window_rows, resulting_target_column='TARGET', numNeighbours=10)

def backtest_test(prices_df: pd.DataFrame):
	all_possible_combinations = CreateAllPossibleCombinations(prices_df)
	np.random.shuffle(all_possible_combinations)

	all_possible_combinations_slice = all_possible_combinations[:3]
	all_possible_combinations_slice = [('close_icp-usdt', 'close_woo-usdt')]
	trade_window_days = 30
	train_window_days = (prices_df.index[-1] - prices_df.index[0]).days - trade_window_days
	# train_window_days = 180
	backtester = Backtester(prices_df=prices_df,
							train_window_days=train_window_days,
							val_window_days=trade_window_days,
							trade_window_days=trade_window_days,
							features_rolling_window_days=10,
							target_rolling_window_days=10,
							all_possible_combinations=all_possible_combinations_slice,
							comovement_detection_type=ComovementType.COINTEGRATION,
							num_target_neighbors=10,
							use_parallelization=False,
							combination_limit=1000,
							trade_limit=1000,
							risk_free_rate=0,
							fees=0.02/100)
	backtester.Run()

if __name__ == '__main__':
	now_str = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
	SetLogging(f'wqu_capstone_{now_str}.log', False)

	prices_df = pd.read_csv('dataset/binance_1h_ohlcv_2021-2025.csv', index_col='date', parse_dates=True)

	# manual_test(prices_df)
	backtest_test(prices_df)