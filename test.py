import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from backtester import Backtester
from startup_helpers import IsLoggingConfigured, ResetLogFileHandler, SetLogging
from top_model import TopModelType
from combinations import CreateAllPossibleCombinations
from comovement import ComovementType, test_cointegration
from feature_engineering import AddFeatures
from spread import AddPolyfitSpread
from target_creation import AddPeakNeighboursTarget, TargetType


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
	# all_possible_combinations_slice = [('close_vet-usdt', 'close_sc-usdt')]

	all_possible_combinations_slice = [('close_algo-usdt', 'close_reef-usdt')]
	# all_possible_combinations_slice = [('close_bat-usdt', 'close_omg-usdt')]
	# all_possible_combinations_slice = [('close_powr-usdt', 'close_algo-usdt'), ('close_troy-usdt', 'close_ach-usdt'), ('close_amp-usdt', 'close_clv-usdt'),
	# 								   ('close_rei-usdt', 'close_algo-usdt'), ('close_voxel-usdt', 'close_algo-usdt'), ('close_amp-usdt', 'close_bico-usdt'),
	# 								   ('close_badger-usdt', 'close_ach-usdt'), ('close_amp-usdt', 'close_celo-usdt'), ('close_rei-usdt', 'close_ach-usdt')]
	trade_window_days = 30 # NOTE: 10 (10%)
	# train_window_days = (prices_df.index[-1] - prices_df.index[0]).days - trade_window_days
	train_window_days = 360 # NOTE: 100 (80%)
	# target_params = {'numNeighbours': 10, 'rolling_window_days': 10}
	# NOTE: best so far
	# NOTE: target_params = {'look_ahead_days': 2, 'reg_points_thresh_frac': 0.75, 'exceedance_thresh_frac': 0.001}
	target_params = {'look_ahead_days': 20, 'reg_points_thresh_frac': 0.75, 'exceedance_thresh_frac': 0.001}
	backtester = Backtester(prices_df=prices_df,
							train_window_days=train_window_days,
							ml_val_window_days=trade_window_days,
							trade_window_days=trade_window_days,
							val_test_split_coef=0.5,
							features_rolling_windows_days_list=[1, 5, 10], # [1, 2, 3]
							all_possible_combinations=all_possible_combinations,
							comovement_detection_type=ComovementType.GC_MI,
							use_parallelization=True,
							combination_limit=1000,
							trade_limit=1000,
							risk_free_rate=0,
							fees=0.1 / 100,
							min_val_net_return=0.1,
							min_val_num_trades=trade_window_days*5,
							num_good_combs_to_choose=300,
							use_top_model=None, # TopModelType.ARIMA is ready to use
							target_type=TargetType.OLS_CLF,
							target_params=target_params,
							close_on_no_signal=True)
	backtester.Run()

def ml_quality_test(prices_df: pd.DataFrame, train_window_days, trade_window_days, num_good_combs_to_choose, desired_num_samples, target_window):
	all_possible_combinations = CreateAllPossibleCombinations(prices_df)
	# np.random.shuffle(all_possible_combinations)
	# all_possible_combinations_slice = all_possible_combinations[:500]

	target_params = {'look_ahead_days': target_window, 'reg_points_thresh_frac': 0.75, 'exceedance_thresh_frac': 0.001}
	backtester = Backtester(prices_df=prices_df,
							train_window_days=train_window_days,
							ml_val_window_days=trade_window_days,
							trade_window_days=trade_window_days,
							val_test_split_coef=0.5,
							features_rolling_windows_days_list=[1, 5, 10],
							all_possible_combinations=all_possible_combinations,
							comovement_detection_type=ComovementType.GC_MI,
							use_parallelization=True,
							combination_limit=1000,
							trade_limit=1000,
							risk_free_rate=0,
							fees=0.1 / 100,
							min_val_net_return=0.1,
							min_val_num_trades=trade_window_days*5,
							num_good_combs_to_choose=num_good_combs_to_choose,
							use_top_model=None, # TopModelType.ARIMA is ready to use
							target_type=TargetType.OLS_CLF,
							target_params=target_params,
							close_on_no_signal=False)

	backtester.MLPredictionQualityTest(desired_num_samples=desired_num_samples)

def run_consecutive_ml_quality_tests():
	now_str = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')

	train_window_days_list = [720, 360, 180, 90, 50]
	trade_window_days = 30
	num_good_combs_to_choose = 200
	desired_num_samples = 5
	target_window_list = [10, 5]
	prices_df = pd.read_csv('dataset/binance_1h_ohlcv_2021-2025.csv', index_col='date', parse_dates=True)

	for target_window in target_window_list:
		for train_window_days in train_window_days_list:
			log_file_name = (f'logs/wqu_capstone_{now_str}_trn{train_window_days}_trd{trade_window_days}_'
							 f'ncombs{num_good_combs_to_choose}_dsmpl{desired_num_samples}_tarwin{target_window}.log')
			if not IsLoggingConfigured():
				SetLogging(log_file_name)
			else:
				ResetLogFileHandler(log_file_name)

			prices_df_copy = prices_df.copy()
			ml_quality_test(prices_df_copy, train_window_days, trade_window_days, num_good_combs_to_choose, desired_num_samples, target_window)
			logging.info('Finished')

if __name__ == '__main__':
	now_str = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
	os.makedirs('logs', exist_ok=True)
	# parallel_logging(f'logs/wqu_capstone_{now_str}.log')

	run_consecutive_ml_quality_tests()
	# TODO: Test run
	# prices_df = prices_df[(prices_df.index >= '2023-02-01') & (prices_df.index <= '2024-07-01')]
	# prices_df = prices_df[(prices_df.index >= '2022-01-01') & (prices_df.index <= '2024-09-01')]

	# manual_test(prices_df)
	# backtest_test(prices_df)
	# logging.info('Finished')