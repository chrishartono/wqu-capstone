from datetime import timedelta

import pandas as pd

from combinations import SearchForGoodCombinations
from comovement import ComovementType
from feature_engineering import AddFeatures
from spread import AddTrainSpread, UpdateSpread
from top_model import rolling_window


class Backtester:

	def __init__(self,
				 prices_df: pd.DataFrame,
				 train_window_days: int,
				 trade_window_days: int,
				 all_possible_combinations: list[tuple[str, str]],
				 comovement_detection_type: ComovementType,
				 use_parallelization: bool):

		self.__prices_df = prices_df
		self.__train_window_days = train_window_days
		self.__all_possible_combinations = all_possible_combinations
		self.__comovement_type = comovement_detection_type
		self.__date_bounds = self.__make_date_bounds(prices_df, train_window_days, trade_window_days)
		self.__n_jobs = -1 if use_parallelization else 1

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
				date_bounds.append((current_bound_date, last_date + train_window, last_date))
			else:
				date_bounds.append((current_bound_date, current_bound_date + train_window, current_bound_date + train_window + trade_window))

			current_bound_date = current_bound_date + trade_window

		return date_bounds

	def __process_single_combination(self, train: pd.DataFrame, test: pd.DataFrame, combination: tuple[str, str]) -> tuple:
		train_spread, coef = AddTrainSpread(train, combination)
		test_spread = UpdateSpread(test, combination, coef)

		train = pd.concat([train, train_spread])
		test = pd.concat([test, test_spread])

		rolling_window_days = int(self.__train_window_days / 2)
		train, test = AddFeatures(train, test, combination, rolling_window_days)
		# TODO: Build targets, train top and bottom models, predict, trade

	def Run(self):

		for start_date, end_train_date, end_test_date in self.__date_bounds:
			train = self.__prices_df[(self.__prices_df.index > start_date) & (self.__prices_df.index <= end_train_date)]
			test = self.__prices_df[(self.__prices_df.index > end_train_date) & (self.__prices_df.index <= end_test_date)]

			good_combinations = SearchForGoodCombinations(train, self.__all_possible_combinations, self.__comovement_type, self.__n_jobs)

			# TODO: Run individual combinations, combine results into portfolio returns


