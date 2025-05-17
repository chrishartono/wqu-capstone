import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from concurrent_log_handler import ConcurrentRotatingFileHandler

from backtester import Backtester
from bottop_prediction import TopModelType
from combinations import CreateAllPossibleCombinations
from comovement import ComovementType, test_cointegration
from feature_engineering import AddFeatures
from spread import AddPolyfitSpread
from target_creation import AddPeakNeighboursSingleColumn

def parallel_logging(name):
	logger = logging.getLogger()
	# Check if handlers are already configured (prevents duplicate handlers)
	if not logger.handlers:
		logger.setLevel(logging.INFO)

		# Set up file handler with concurrency support
		file_handler = ConcurrentRotatingFileHandler(name, mode='a', maxBytes=1024 * 1024, backupCount=5)
		file_format = logging.Formatter('%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
		file_handler.setFormatter(file_format)
		logger.addHandler(file_handler)

		# Set up console handler
		console_handler = logging.StreamHandler()
		console_format = logging.Formatter('%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
		console_handler.setFormatter(console_format)
		logger.addHandler(console_handler)

	return logger

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
	# np.random.shuffle(all_possible_combinations)

	# all_possible_combinations_slice = all_possible_combinations[:1000]
	# all_possible_combinations_slice = [('close_powr-usdt', 'close_algo-usdt')]
	# all_possible_combinations_slice = [('close_powr-usdt', 'close_algo-usdt'), ('close_troy-usdt', 'close_ach-usdt'), ('close_amp-usdt', 'close_clv-usdt'),
	# 								   ('close_rei-usdt', 'close_algo-usdt'), ('close_voxel-usdt', 'close_algo-usdt'), ('close_amp-usdt', 'close_bico-usdt'),
	# 								   ('close_badger-usdt', 'close_ach-usdt'), ('close_amp-usdt', 'close_celo-usdt'), ('close_rei-usdt', 'close_ach-usdt')]
	trade_window_days = 60
	# train_window_days = (prices_df.index[-1] - prices_df.index[0]).days - trade_window_days
	train_window_days = 360
	backtester = Backtester(prices_df=prices_df,
							train_window_days=train_window_days,
							ml_val_window_days=trade_window_days,
							trade_window_days=trade_window_days,
							val_test_split_coef=0.5,
							features_rolling_window_days=10,
							target_rolling_window_days=10,
							all_possible_combinations=all_possible_combinations,
							comovement_detection_type=ComovementType.GC_MI,
							num_target_neighbors=10,
							use_parallelization=True,
							combination_limit=1000,
							trade_limit=1000,
							risk_free_rate=0,
							fees=0.1 / 100,
							min_val_net_return=0.1,
							min_val_num_trades=trade_window_days,
							use_top_model=None) # TopModelType.ARIMA is ready to use
	backtester.Run()

if __name__ == '__main__':
	now_str = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
	os.makedirs('logs', exist_ok=True)
	SetLogging(f'logs/wqu_capstone_{now_str}.log', False)
	# parallel_logging(f'logs/wqu_capstone_{now_str}.log')

	prices_df = pd.read_csv('dataset/binance_1h_ohlcv_2021-2025.csv', index_col='date', parse_dates=True)

	# manual_test(prices_df)
	backtest_test(prices_df)
	logging.info('Finished')