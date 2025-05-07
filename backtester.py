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
from utils.helpers import DaysWindowToPeriods, SemiStd


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
				 trade_limit: float,
				 risk_free_rate: float):

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
		self.__risk_free_rate = risk_free_rate

		self.__annualized_multiplier = np.sqrt(24 * 365)

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

	def __update_stats(self, prices, combination_pos, pair_cash_pos, pair_pos, pair_mtm, combination_exposure_trades, last_pair_cash_pos, last_pair_pos, coef,
					   i):
		combination_pos.append(combination_exposure_trades)
		pair_cash_pos[0].append(last_pair_cash_pos[0])
		pair_cash_pos[1].append(last_pair_cash_pos[1])
		pair_pos[0].append(last_pair_pos[0])
		pair_pos[1].append(last_pair_pos[1])
		# Sign is already in last_pair_pos, so coef is taken as abs
		pair_mtm[0].append(last_pair_cash_pos[0] + last_pair_pos[0] * prices[0][i] * abs(coef[0]))
		pair_mtm[1].append(last_pair_cash_pos[1] + last_pair_pos[1] * prices[1][i] * abs(coef[1]))

	def __finalize_backtest(self,
							prices,
							combination_pos,
							pair_cash_pos,
							pair_pos,
							pair_mtm,
							combination_exposure_trades,
							last_pair_cash_pos,
							last_pair_pos,
							coef,
							i):
		if combination_exposure_trades > 0:
			pair_cash_pos[0].append(last_pair_cash_pos[0] - abs(last_pair_pos[0]) * prices[0][i] * coef[0])
			pair_cash_pos[1].append(last_pair_cash_pos[1] - abs(last_pair_pos[1]) * prices[1][i] * coef[1])
		elif combination_exposure_trades < 0:
			pair_cash_pos[0].append(last_pair_cash_pos[0] + abs(last_pair_pos[0]) * prices[0][i] * coef[0])
			pair_cash_pos[1].append(last_pair_cash_pos[1] + abs(last_pair_pos[1]) * prices[1][i] * coef[1])

		combination_pos.append(0)
		pair_pos[0].append(0)
		pair_pos[1].append(0)
		pair_mtm[0].append(pair_cash_pos[0][-1])
		pair_mtm[1].append(pair_cash_pos[1][-1])

	def __calc_metrics(self, combination_mtm: list[float], trading_days: float):
		mtm = np.array(combination_mtm)

		net_return = (mtm[-1] - mtm[0]) / mtm[0]
		annualized_net_return = net_return / trading_days * 365

		mtm_returns = mtm[1:] / mtm[:-1] - 1
		mean_mtm_return = np.mean(mtm_returns)
		std_mtm_return = np.std(mtm_returns)
		semi_std_mtm_return = SemiStd(mtm_returns)
		sharpe = (mean_mtm_return - self.__risk_free_rate) / std_mtm_return * self.__annualized_multiplier
		sortino = (mean_mtm_return - self.__risk_free_rate) / semi_std_mtm_return * self.__annualized_multiplier

		# accumulate max value and subtract actual value. doing this we get the maximum fall
		runningDD = np.maximum.accumulate(mtm) - mtm
		pointDD = runningDD.argmax()
		DD = runningDD[pointDD]
		peak = max(mtm[:pointDD]) if pointDD > 0 else 0
		maxDD = DD / peak if peak != 0 else 0

		cash_netprofit = mtm[-1] - mtm[0]
		recoveryFactor = cash_netprofit / maxDD if maxDD != 0 else 0

		metrics = {'annualized_net_return': annualized_net_return,
				   'sharpe'               : sharpe,
				   'sortino'              : sortino,
				   'maxDD'                : maxDD,
				   'recoveryFactor'       : recoveryFactor}
		return metrics

	def __trading_logic(self, combination: tuple[str, str], test: pd.DataFrame, preds: np.ndarray, coint_vector: PhillipsOuliarisTestResults):
		pair0 = combination[0]
		pair1 = combination[1]
		coef_orig = [coint_vector[pair0], coint_vector[pair1]]
		prices = [test[pair0].to_list(), test[pair1].to_list()]

		combination_pos = []
		pair_cash_pos = [[], []]
		pair_pos = [[], []]
		pair_mtm = [[], []]
		coef_history = [[], []]

		# Max number of trades we can open according to backtest settings: trade_limit and combination_limit
		max_exposure = int(self.__combination_limit / self.__trade_limit)

		# Current strategy exposure in number of open trades
		combination_exposure_trades = 0

		# Latest cumulative cash position for each pair. If we buy pair, we spend cash, so cash position is negative. And vice versa.
		last_pair_cash_pos = [0, 0]

		# Lates cumulative pair position in trades. Buy once, get position=1.
		last_pair_pos = [0, 0]

		for i, prediction in enumerate(preds):

			# Here we calculate the total margin for an open combination position based on current prices and cointegration coefficients.
			# Margin value equals total abs cash flow. But we have a trade_limit setting, so we have to adjust our trade coefficients accordingly.
			face_value_margin = abs(prices[0][i] * coef_orig[0]) + abs(prices[1][i] * coef_orig[1])
			# So if our face_value_margin was > trade_limit, we will shrink our coefs so that total trade margin does not exceed trade_limit
			coef_adjustment = self.__trade_limit / face_value_margin
			# coef = [coef_orig[0] * coef_adjustment, coef_orig[1] * coef_adjustment]
			coef = [coef_orig[0], coef_orig[1]]

			if prediction == SignalTypes.BUY.value:
				if combination_exposure_trades + 1 <= max_exposure:
					# We BUY spread. Increase strategy exposure by 1 trade
					combination_exposure_trades += 1

					# If we BUY spread, it means that we use coefs with signs as they were given by cointegration.
					# Say we had cointegration coefs [0.8, -3.8]. Buying spread means buying 0.8 pair0 and selling 3.8 pair1.
					# But cash flow has opposite sign. So we have spent cash to buy 0.8 pair0 and earned  after selling 3.8 pair1.
					# To represent this we subtract trade volume from last cash_pos. If coef > 0, it means we should buy, then we subtract trade volume.
					# If coef < 0, it means we should sell, then minus * minus gives plus => we add trade volume.
					# And as this is a cumulative cash_pos, we add it to the previous one.
					last_pair_cash_pos = [last_pair_cash_pos[0] - prices[0][i] * coef[0], last_pair_cash_pos[1] - prices[1][i] * coef[1]]
					# Positions counted as number of trades. Add to the previous.
					last_pair_pos = [last_pair_pos[0] + np.sign(coef[0]), last_pair_pos[1] + np.sign(coef[1])]

			elif prediction == SignalTypes.SELL.value:
				if combination_exposure_trades - 1 >= -max_exposure:
					combination_exposure_trades -= 1
					# Opposite here. Flip the signs.
					last_pair_cash_pos = [last_pair_cash_pos[0] + prices[0][i] * coef[0], last_pair_cash_pos[1] + prices[1][i] * coef[1]]
					last_pair_pos = [last_pair_pos[0] - np.sign(coef[0]), last_pair_pos[1] - np.sign(coef[1])]

			coef_history[0].append(coef[0])
			coef_history[1].append(coef[1])
			# Add all last values to the lists of running statistics.
			self.__update_stats(prices,
								combination_pos,
								pair_cash_pos,
								pair_pos,
								pair_mtm,
								combination_exposure_trades,
								last_pair_cash_pos,
								last_pair_pos,
								coef,
								i)

		self.__finalize_backtest(prices,
								 combination_pos,
								 pair_cash_pos,
								 pair_pos,
								 pair_mtm,
								 combination_exposure_trades,
								 last_pair_cash_pos,
								 last_pair_pos,
								 coef,
								 i)

		res_df = pd.DataFrame(prices[0], columns=[pair0])
		res_df['coef0'] = coef_history[0]
		res_df['cash_pos0'] = pair_cash_pos[0][:-1]
		res_df['pos0'] = pair_pos[0][:-1]
		res_df['mtm0'] = pair_mtm[0][:-1]
		res_df[pair1] = prices[1]
		res_df['coef1'] = coef_history[1]
		res_df['cash_pos1'] = pair_cash_pos[1][:-1]
		res_df['pos1'] = pair_pos[1][:-1]
		res_df['mtm1'] = pair_mtm[1][:-1]


		trading_days = (test.index[-1] - test.index[0]).days
		combination_mtm = [t[0] + t[1] + self.__combination_limit for t in zip(*pair_mtm)]
		metrics = self.__calc_metrics(combination_mtm, trading_days)

		updated_index = test.index.append(test.index[-1] + timedelta(seconds=10))
		stats_df = pd.DataFrame(combination_mtm, index=updated_index, columns=['combination_mtm'])
		stats_df[f'combination_pos'] = combination_pos

		return stats_df, metrics

	def Run(self):

		for start_date, end_train_date, end_test_date in self.__date_bounds:
			all_train = self.__prices_df[(self.__prices_df.index > start_date) & (self.__prices_df.index <= end_train_date)]
			all_test = self.__prices_df[(self.__prices_df.index > end_train_date) & (self.__prices_df.index <= end_test_date)]

			good_combinations = SearchForGoodCombinations(all_train, self.__all_possible_combinations, self.__comovement_type, self.__n_jobs)
			data_tuples = self.__prepare_all_combination_datas(good_combinations, all_train, all_test)

			for comb_train_set, comb_test_set, combination, coint_vector in data_tuples:
				preds = Train(comb_train_set, comb_test_set, combination, self.__val_window_days)
				stats_df, metrics = self.__trading_logic(combination, comb_test_set, preds, coint_vector)
# TODO: Run individual combinations, combine results into portfolio returns
