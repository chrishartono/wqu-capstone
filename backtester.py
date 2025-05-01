import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from arch.unitroot._phillips_ouliaris import PhillipsOuliarisTestResults
from joblib import Parallel, delayed
from tqdm import tqdm

from bottop_prediction import Train
from combinations import SearchForGoodCombinations
from comovement import ComovementType
from feature_engineering import AddFeatures
from spread import AddCointCoefSpread
from target_creation import AddPeakNeighboursSingleColumn
from utils.data_structures import SignalTypes
from utils.helpers import DaysWindowToPeriods


class Backtester:

	def __init__(self,
				 prices_df: pd.DataFrame,
				 train_window_days: int,
				 val_window_days: int,
				 trade_window_days: int,
				 features_rolling_window_days: int,
				 target_rolling_window_days: int,
				 all_possible_combinations: list[tuple[str, str]],
				 comovement_detection_type: ComovementType,
				 num_target_neighbors: int,
				 use_parallelization: bool,
				 combination_limit: float,
				 trade_limit: float):

		self.__prices_df = prices_df
		self.__train_window_days = train_window_days
		self.__val_window_days = val_window_days
		self.__features_rolling_window_days = features_rolling_window_days
		self.__target_rolling_window_days = target_rolling_window_days
		self.__all_possible_combinations = all_possible_combinations
		self.__comovement_type = comovement_detection_type
		self.__num_target_neighbors = num_target_neighbors
		self.__date_bounds = self.__make_date_bounds(prices_df, train_window_days, trade_window_days)
		self.__n_jobs = -1 if use_parallelization else 1
		self.__combination_limit = combination_limit
		self.__trade_limit = trade_limit

	@staticmethod
	def __make_date_bounds(prices_df: pd.DataFrame, train_window_days: int, trade_window_days: int):
		"""
		This function creates boundaries for data train/test slices for walkforward backtest in format (start_train_date, end_train_date, end_test_date)

		:param prices_df: Pandas DataFrame with 2 columns (one for each time series).
		:param train_window_days: Number of days for train set.
		:param trade_window_days: Number of days for test set.
		:return: List of tuples of datetime values for boundaries.
		"""

		last_date = prices_df.index[-1]

		# To make sure that the first row is included
		current_bound_date = prices_df.index[0] - timedelta(seconds=10)

		train_window = timedelta(days=train_window_days)
		trade_window = timedelta(days=trade_window_days)

		date_bounds = []

		while current_bound_date + train_window <= last_date:

			if current_bound_date + train_window + trade_window > last_date:
				date_bounds.append((current_bound_date, current_bound_date + train_window, last_date))
			else:
				date_bounds.append((current_bound_date, current_bound_date + train_window, current_bound_date + train_window + trade_window))

			current_bound_date = current_bound_date + trade_window

		return date_bounds

	def __prepare_combination_data(self, train: pd.DataFrame, test: pd.DataFrame, combination: tuple[str, str], coint_vector: PhillipsOuliarisTestResults):
		train = AddCointCoefSpread(train, combination, coint_vector)
		test = AddCointCoefSpread(test, combination, coint_vector)

		train, test = AddFeatures(train, test, combination, self.__features_rolling_window_days)
		data = pd.concat([train, test], axis=0)

		window_rows = DaysWindowToPeriods(data, self.__target_rolling_window_days)
		data = AddPeakNeighboursSingleColumn(data,
											 combination,
											 target_col='spread',
											 period=window_rows,
											 resulting_target_column='TARGET',
											 numNeighbours=self.__num_target_neighbors)

		train = data.iloc[:len(train)]
		test = data.iloc[len(train):]

		return train, test, combination, coint_vector
		# TODO: Build targets, train top and bottom models, predict, trade

	def __prepare_all_combination_datas(self,
										good_combinations: list[tuple[tuple[str, str], PhillipsOuliarisTestResults]],
										train: pd.DataFrame,
										test: pd.DataFrame):
		logging.info(f'Start features and target preparations for {len(good_combinations)} combinations on '
					 f'train set from {train.index[0]} to {train.index[-1]} and '
					 f'test set from {test.index[0]} to {test.index[-1]}')

		params = []
		for comb, coint_vector in good_combinations:
			pair1 = comb[0].split('_')[1]
			pair2 = comb[1].split('_')[1]
			comb_columns = [col for col in train.columns if pair1 in col or pair2 in col]
			params.append((train[comb_columns], test[comb_columns], comb, coint_vector))

		results = (Parallel(n_jobs=self.__n_jobs, prefer="processes")
				   (delayed(self.__prepare_combination_data)(*p) for p in tqdm(params, total=len(params), desc="Train data preparations:")))

		logging.info(f'Finally got {len(results)} data tuples')

		return results

	def _update_stats(self, prices, pair_cash_pos, pair_pos, pair_mtm, last_pair_cash_pos, last_pair_pos, coef, i):
		pair_cash_pos[0].append(last_pair_cash_pos[0])
		pair_cash_pos[1].append(last_pair_cash_pos[1])
		pair_pos[0].append(last_pair_pos[0])
		pair_pos[1].append(last_pair_pos[1])
		pair_mtm[0].append(last_pair_cash_pos[0] + last_pair_pos[0] * prices[0][i] * coef[0])
		pair_mtm[1].append(last_pair_cash_pos[1] + last_pair_pos[1] * prices[1][i] * coef[1])

	def __trading_logic(self, combination: tuple[str, str], test: pd.DataFrame, preds: np.ndarray, coint_vector: PhillipsOuliarisTestResults):
		pair0 = combination[0]
		pair1 = combination[1]
		coef = [coint_vector[pair0], coint_vector[pair1]]
		prices = [test[pair0].to_list(), test[pair1].to_list()]

		pair_cash_pos = [[], []]
		pair_pos = [[], []]
		pair_mtm = [[], []]

		comb_pos = 0
		last_pair_cash_pos = [0, 0]
		last_pair_pos = [0, 0]
		for i, prediction in enumerate(preds):

			if prediction == SignalTypes.NONE.value:
				self._update_stats(prices, pair_cash_pos, pair_pos, pair_mtm, last_pair_cash_pos, last_pair_pos, coef, i)
				continue

			if prediction == SignalTypes.BUY.value:
				if comb_pos + self.__trade_limit > self.__combination_limit:
					self._update_stats(prices, pair_cash_pos, pair_pos, pair_mtm, last_pair_cash_pos, last_pair_pos, coef, i)
					continue

				comb_pos += self.__trade_limit
				last_pair_cash_pos


		pass

	def Run(self):

		for start_date, end_train_date, end_test_date in self.__date_bounds:
			all_train = self.__prices_df[(self.__prices_df.index > start_date) & (self.__prices_df.index <= end_train_date)]
			all_test = self.__prices_df[(self.__prices_df.index > end_train_date) & (self.__prices_df.index <= end_test_date)]

			good_combinations = SearchForGoodCombinations(all_train, self.__all_possible_combinations, self.__comovement_type, self.__n_jobs)
			data_tuples = self.__prepare_all_combination_datas(good_combinations, all_train, all_test)

			for comb_train_set, comb_test_set, combination, coint_vector in data_tuples:
				preds = Train(comb_train_set, comb_test_set, combination, self.__val_window_days)
			# TODO: Run individual combinations, combine results into portfolio returns


