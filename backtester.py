import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas
import pandas as pd
from arch.unitroot._phillips_ouliaris import PhillipsOuliarisTestResults
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from pandas import DatetimeIndex
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
				 risk_free_rate: float,
				 fees: float):

		self.__backtest_id = str(uuid.uuid4())
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
		self.__fees = fees

		self.__annualized_multiplier = np.sqrt(24 * 365)

		self.__portfolio_df = None
		self.__stats_by_comb = {}

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

	def prepare_combination_data(self, train: pd.DataFrame, test: pd.DataFrame, combination: tuple[str, str], coint_vector: PhillipsOuliarisTestResults):
		try:
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
		except:
			return None, None, combination, coint_vector

		return train, test, combination, coint_vector

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
				   (delayed(self.prepare_combination_data)(*p) for p in tqdm(params, total=len(params), desc="Train data preparations:")))

		results = [tup for tup in results if tup[0] is not None and tup[1] is not None]
		logging.info(f'Finally got {len(results)} data tuples')

		return results

	def __update_stats(self, prices, combination_pos, pair_cash_pos, pair_pos, pair_mtm, combination_exposure_trades, last_pair_cash_pos, last_pair_pos, coef,
					   i, last_fees):
		combination_pos.append(combination_exposure_trades)
		pair_cash_pos[0].append(last_pair_cash_pos[0])
		pair_cash_pos[1].append(last_pair_cash_pos[1])
		pair_pos[0].append(last_pair_pos[0])
		pair_pos[1].append(last_pair_pos[1])
		# Sign is already in last_pair_pos, so coef is taken as abs
		pair_mtm[0].append(last_pair_cash_pos[0] + last_pair_pos[0] * prices[0][i] * abs(coef[0]) - last_fees[0])
		pair_mtm[1].append(last_pair_cash_pos[1] + last_pair_pos[1] * prices[1][i] * abs(coef[1]) - last_fees[1])

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
		if combination_exposure_trades != 0:
			pair_cash_pos[0].append(last_pair_cash_pos[0] + last_pair_pos[0] * prices[0][i] * abs(coef[0]))
			pair_cash_pos[1].append(last_pair_cash_pos[1] + last_pair_pos[1] * prices[1][i] * abs(coef[1]))
		# if combination_exposure_trades > 0:
		# 	pair_cash_pos[0].append(last_pair_cash_pos[0] + abs(last_pair_pos[0]) * prices[0][i] * coef[0])
		# 	pair_cash_pos[1].append(last_pair_cash_pos[1] - abs(last_pair_pos[1]) * prices[1][i] * coef[1])
		# elif combination_exposure_trades < 0:
		# 	pair_cash_pos[0].append(last_pair_cash_pos[0] - abs(last_pair_pos[0]) * prices[0][i] * coef[0])
		# 	pair_cash_pos[1].append(last_pair_cash_pos[1] + abs(last_pair_pos[1]) * prices[1][i] * coef[1])

		combination_pos.append(0)
		pair_pos[0].append(0)
		pair_pos[1].append(0)
		pair_mtm[0].append(pair_cash_pos[0][-1])
		pair_mtm[1].append(pair_cash_pos[1][-1])

	def __calc_metrics(self, mtm: np.ndarray, trading_days: float):
		net_return = (mtm[-1] - mtm[0]) / mtm[0]
		annualized_net_return = net_return / trading_days * 365

		mtm_returns = mtm[1:] / mtm[:-1] - 1
		mean_mtm_return = np.mean(mtm_returns)
		std_mtm_return = np.std(mtm_returns)
		semi_std_mtm_return = SemiStd(mtm_returns)
		sharpe = (mean_mtm_return - self.__risk_free_rate) / std_mtm_return * self.__annualized_multiplier if std_mtm_return != 0 else 0
		sortino = (mean_mtm_return - self.__risk_free_rate) / semi_std_mtm_return * self.__annualized_multiplier if semi_std_mtm_return != 0 else 0

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
		prices = [test[pair0].to_numpy(), test[pair1].to_numpy()]

		# Here we calculate the total margin for an open combination position based on current prices and cointegration coefficients.
		# Margin value equals total abs cash flow. But we have a trade_limit setting, so we have to adjust our trade coefficients accordingly.
		# face_value_margin = abs(prices[0][0] * coef_orig[0]) + abs(prices[1][0] * coef_orig[1])
		# So if our face_value_margin was > trade_limit, we will shrink our coefs so that total trade margin does not exceed trade_limit
		# coef_adjustment = self.__trade_limit / face_value_margin

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

		# Latest cumulative pair position in trades. Buy once, get position=1.
		last_pair_pos = [0, 0]

		for i, prediction in enumerate(preds):
			# coef = [coef_orig[0] * coef_adjustment, coef_orig[1] * coef_adjustment]
			coef = [coef_orig[0], coef_orig[1]]

			last_fees = [0, 0]

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
					last_fees = [abs(prices[0][i] * coef[0] * self.__fees), abs(prices[1][i] * coef[1] * self.__fees)]

			elif prediction == SignalTypes.SELL.value:
				if combination_exposure_trades - 1 >= -max_exposure:
					combination_exposure_trades -= 1
					# Opposite here. Flip the signs.
					last_pair_cash_pos = [last_pair_cash_pos[0] + prices[0][i] * coef[0], last_pair_cash_pos[1] + prices[1][i] * coef[1]]
					last_pair_pos = [last_pair_pos[0] - np.sign(coef[0]), last_pair_pos[1] - np.sign(coef[1])]
					last_fees = [abs(prices[0][i] * coef[0] * self.__fees), abs(prices[1][i] * coef[1] * self.__fees)]

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
								i,
								last_fees)

		# self.__finalize_backtest(prices,
		# 						 combination_pos,
		# 						 pair_cash_pos,
		# 						 pair_pos,
		# 						 pair_mtm,
		# 						 combination_exposure_trades,
		# 						 last_pair_cash_pos,
		# 						 last_pair_pos,
		# 						 coef,
		# 						 i)

		# res_df = pd.DataFrame(prices[0], columns=[pair0])
		# res_df['coef0'] = coef_history[0]
		# res_df['cash_pos0'] = pair_cash_pos[0][:-1]
		# res_df['pos0'] = pair_pos[0][:-1]
		# res_df['mtm0'] = pair_mtm[0][:-1]
		# res_df[pair1] = prices[1]
		# res_df['coef1'] = coef_history[1]
		# res_df['cash_pos1'] = pair_cash_pos[1][:-1]
		# res_df['pos1'] = pair_pos[1][:-1]
		# res_df['mtm1'] = pair_mtm[1][:-1]

		trading_days = (test.index[-1] - test.index[0]).days
		combination_mtm_0based = [t[0] + t[1] for t in zip(*pair_mtm)]
		capital_usage = [abs(t[0]) + abs(t[1]) for t in zip(*pair_cash_pos)]
		max_capital_usage = max(capital_usage)

		# If we made no trades, max_capital_usage=0. But combination_mtm_max_capital_based list will also contain only zeros then.
		# And in the end we will have issues with calculating metrics
		max_capital_usage = max(max_capital_usage, 0.01)
		combination_mtm_max_capital_based = [mtm + max_capital_usage for mtm in combination_mtm_0based]

		# hasbad = np.any([(np.isnan(mtm) or mtm==0) for mtm in combination_mtm_max_capital_based])
		# if hasbad:
		# 	logging.error(f'CAUTION!!! {combination}. coint_vector={coint_vector} max_capital_usage={max_capital_usage}')
		# 	exit(1)
		# shifted_last_idx_value = test.index[-1] + timedelta(seconds=10)
		# new_idx_value_as_list = DatetimeIndex([shifted_last_idx_value], dtype='datetime64[ns]', freq=None)
		# updated_index = test.index.append(new_idx_value_as_list)

		stats_df = pd.DataFrame(combination_mtm_max_capital_based, index=test.index, columns=['mtm'])

		coef_history_arrays = [np.array(coef_history[0]), np.array(coef_history[1])]
		stats_df['spread'] = prices[0] * coef_history_arrays[0] + prices[1] * coef_history_arrays[1]
		stats_df[f'pos'] = combination_pos
		stats_df[f'mtm_returns'] = (stats_df['mtm'] - stats_df['mtm'].shift(1)) / abs(stats_df['mtm'].shift(1))
		stats_df[f'mtm_returns'].replace([np.inf, -np.inf], np.nan, inplace=True)
		stats_df[f'mtm_returns'].fillna(0, inplace=True)

		metrics = self.__calc_metrics(np.array(combination_mtm_max_capital_based), trading_days)
		# mtm_returns_cumprod = (stats_df[f'mtm_returns'] + 1).cumprod()
		# test_metrics_cumprod = self.__calc_metrics(mtm_returns_cumprod.to_numpy(), trading_days)

		return stats_df, metrics

	def __combine_results(self, comb_stats_tups: list[tuple[tuple[str, str], pandas.DataFrame]]):

		individual_dfs = []
		for combination, stats_df in comb_stats_tups:
			if combination not in self.__stats_by_comb:
				self.__stats_by_comb[combination] = stats_df
			else:
				old_stats_df = self.__stats_by_comb[combination]
				self.__stats_by_comb[combination] = pd.concat([old_stats_df, stats_df], axis=0)

			individual_dfs.append(stats_df['mtm_returns'])

		iteration_portfolio_df = pd.concat(individual_dfs, axis=1)

		# We assume equal capital allocations across all combinations
		iteration_portfolio_df['portfolio_returns'] = iteration_portfolio_df.mean(axis=1)
		iteration_portfolio_df['active_combinations'] = len(individual_dfs)

		if self.__portfolio_df is None:
			self.__portfolio_df = iteration_portfolio_df[['portfolio_returns', 'active_combinations']]
		else:
			self.__portfolio_df = pd.concat([self.__portfolio_df, iteration_portfolio_df[['portfolio_returns', 'active_combinations']]], axis=0)

	def __save_plot(self, pair0: str, pair1: str, stats_df: pd.DataFrame, metrics: dict, plot_path: str, save_file_name: str):
		nrows = 1
		if 'pos' in stats_df.columns: nrows += 1
		if 'active_combinations' in stats_df.columns: nrows += 1

		fig, ax = plt.subplots(nrows, ncols=1, figsize=(35, 25))
		row_ax = 0
		ax[row_ax].plot(stats_df.index, stats_df['cumprod_mtm_returns'], 'm-')
		ax[row_ax].set_xlabel('Date')
		ax[row_ax].set_ylabel('Cumulative returns', color='m')
		ax[row_ax].set_title(f'{pair0}-{pair1}. Sharpe={metrics["sharpe"]:.4f} Annualized return={metrics["annualized_net_return"]:.4f}')

		if 'spread' in stats_df.columns:
			twin_ax0 = ax[row_ax].twinx()
			twin_ax0.plot(stats_df.index, stats_df['spread'], 'b-')

		if 'pos' in stats_df.columns:
			row_ax += 1
			ax[row_ax].plot(stats_df.index, stats_df['pos'], 'b-')
			ax[row_ax].set_xlabel('Date')
			ax[row_ax].set_ylabel('Position', color='b')

		if 'active_combinations' in stats_df.columns:
			row_ax += 1
			ax[row_ax].plot(stats_df.index, stats_df['active_combinations'], 'b-')
			ax[row_ax].set_xlabel('Date')
			ax[row_ax].set_ylabel('Active combinations', color='b')

		fig.savefig(os.path.join(plot_path, f'{save_file_name}.png'), dpi=300)
		plt.close()

	def __save_all_resulting_metrics(self):
		all_combined_values = []
		now = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
		main_path = f'results/{now}_{self.__backtest_id}'
		plot_path = f'{main_path}/plots'
		equity_path = f'{main_path}/equity'
		os.makedirs(main_path, exist_ok=True)
		os.makedirs(plot_path, exist_ok=True)
		os.makedirs(equity_path, exist_ok=True)

		for combination, stats_df in self.__stats_by_comb.items():
			pair0 = combination[0]
			pair1 = combination[1]

			stats_df['cumprod_mtm_returns'] = (stats_df[f'mtm_returns'] + 1).cumprod()
			trading_days = (stats_df.index[-1] - stats_df.index[0]).days
			metrics = self.__calc_metrics(stats_df['cumprod_mtm_returns'].to_numpy(), trading_days)

			save_file_name = f'{pair0}_{pair1}'

			stats_df[['cumprod_mtm_returns', 'pos']].to_csv(os.path.join(equity_path, f'{save_file_name}.csv'), index=True, index_label='date')

			self.__save_plot(pair0, pair1, stats_df, metrics, plot_path, save_file_name)

		if self.__portfolio_df is not None and len(self.__portfolio_df) > 0:
			self.__portfolio_df['cumprod_mtm_returns'] = (self.__portfolio_df['portfolio_returns'] + 1).cumprod()
			trading_days = (self.__portfolio_df.index[-1] - self.__portfolio_df.index[0]).days
			portfolio_metrics = self.__calc_metrics(self.__portfolio_df['cumprod_mtm_returns'].to_numpy(), trading_days)
			self.__save_plot('portfolio', 'portfolio', self.__portfolio_df, portfolio_metrics, plot_path, 'portfolio_plot')

	def Run(self):

		for start_date, end_train_date, end_test_date in self.__date_bounds:
			all_train = self.__prices_df[(self.__prices_df.index > start_date) & (self.__prices_df.index <= end_train_date)]
			all_test = self.__prices_df[(self.__prices_df.index > end_train_date) & (self.__prices_df.index <= end_test_date)]

			good_combinations = SearchForGoodCombinations(all_train, self.__all_possible_combinations, self.__comovement_type, self.__n_jobs)
			data_tuples = self.__prepare_all_combination_datas(good_combinations, all_train, all_test)

			comb_stats_tups = []
			for comb_train_set, comb_test_set, combination, coint_vector in data_tuples:
				preds = Train(comb_train_set, comb_test_set, combination, self.__val_window_days)
				stats_df, metrics = self.__trading_logic(combination, comb_test_set, preds, coint_vector)

				comb_stats_tups.append((combination, stats_df))

			self.__combine_results(comb_stats_tups)

		self.__save_all_resulting_metrics()
