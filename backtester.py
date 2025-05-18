import gc
import logging
import os
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas
import pandas as pd
from arch.unitroot._phillips_ouliaris import PhillipsOuliarisTestResults

import joblib.externals.loky

joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(3e12)
from joblib import Parallel, delayed

from matplotlib import pyplot as plt
from tqdm import tqdm

from bottop_prediction import Predict, Train, TopModelType
from combinations import SearchForGoodCombinations
from comovement import ComovementType
from feature_engineering import AddFeatures
from spread import AddCointCoefSpread
from target_creation import AddClassificationOLSTarget, AddPeakNeighboursTarget, TargetType
from utils.data_structures import SignalTypes
from utils.helpers import DaysWindowToPeriods, SemiStd


class Backtester:

	def __init__(self,
				 prices_df: pd.DataFrame,
				 train_window_days: int,
				 ml_val_window_days: int,
				 trade_window_days: int,
				 val_test_split_coef: float,
				 features_rolling_window_days: int,
				 target_rolling_window_days: int,
				 all_possible_combinations: list[tuple[str, str]],
				 comovement_detection_type: ComovementType,
				 use_parallelization: bool,
				 combination_limit: float,
				 trade_limit: float,
				 risk_free_rate: float,
				 fees: float,
				 min_val_net_return: float,
				 min_val_num_trades: int,
				 num_good_combs_to_choose: int,
				 use_top_model: TopModelType,
				 target_type: TargetType,
				 target_params: dict):

		self.__backtest_id = str(uuid.uuid4())
		self.__prices_df = prices_df
		self.__train_window_days = train_window_days
		self.__ml_val_window_days = ml_val_window_days
		self.__features_rolling_window_days = features_rolling_window_days
		self.__target_rolling_window_days = target_rolling_window_days
		self.__all_possible_combinations = all_possible_combinations
		self.__comovement_type = comovement_detection_type
		self.__date_bounds = self.__make_date_bounds(prices_df, train_window_days, trade_window_days, val_test_split_coef)
		self.__n_jobs = -1 if use_parallelization else 1
		self.__combination_limit = combination_limit
		self.__trade_limit = trade_limit
		self.__risk_free_rate = risk_free_rate
		self.__fees = fees
		self.__min_val_net_return = min_val_net_return
		self.__min_val_num_trades = min_val_num_trades
		self.__num_good_combs_to_choose = num_good_combs_to_choose

		self.__target_type = target_type
		self.__target_params = target_params
		if target_type == TargetType.PEAK_NEIGHBOURS_CLF:
			self.__AddTargetFunc = AddPeakNeighboursTarget
		elif target_type == TargetType.OLS_CLF:
			self.__AddTargetFunc = AddClassificationOLSTarget
		else:
			raise Exception(f'Unknown target type: {target_type}')


		self.__annualized_multiplier = np.sqrt(24 * 365)

		self.__portfolio_df = None
		self.__stats_by_comb = {}

		self.__use_top_model = use_top_model

	@staticmethod
	def __make_date_bounds_no_val(prices_df: pd.DataFrame, train_window_days: int, trade_window_days: int):
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

	def __make_date_bounds(self, prices_df: pd.DataFrame, train_window_days: int, trade_window_days: int, val_test_split_coef: float):
		"""
		This function creates boundaries for data train/test slices for walkforward backtest in format (start_train_date, end_train_date, end_test_date)

		:param prices_df: Pandas DataFrame with 2 columns (one for each time series).
		:param train_window_days: Number of days for train set.
		:param trade_window_days: Number of days for test set.
		:param val_test_split_coef: Determines VAL set size to VAL+TEST combined set size
		:return: List of tuples of datetime values for boundaries.
		"""

		last_date = prices_df.index[-1]
		current_bound_date = prices_df.index[0] - timedelta(seconds=10)  # To make sure that the first row is included
		val_test_days = trade_window_days / (1 - val_test_split_coef)

		# Here we assume that wf_window_days >> val_test_days
		train_window = timedelta(days=train_window_days)
		trade_window = timedelta(days=trade_window_days)
		val_test_delta = timedelta(days=val_test_days)
		test_delta = trade_window

		date_bounds = []
		# Iterate until there is not enough data to have at least val_test_days for train and val_test_days for val and test
		while current_bound_date + train_window - test_delta <= last_date:

			if current_bound_date + train_window <= last_date:  # The whole wf_window fits before last_date
				date_bounds.append((current_bound_date, current_bound_date + train_window - val_test_delta, current_bound_date + train_window - test_delta,
									current_bound_date + train_window))
			else:  # Only 2 val_test_days windows fit before last_date
				date_bounds.append((
						current_bound_date, current_bound_date + train_window - val_test_delta, current_bound_date + train_window - test_delta, last_date))
				break

			current_bound_date = current_bound_date + trade_window

		return date_bounds

	def prepare_combination_data(self, data: pd.DataFrame, combination: tuple[str, str], coint_vector: PhillipsOuliarisTestResults):
		try:
			# logging.info(f'Start adding spread for {combination}')
			data = AddCointCoefSpread(data, combination, coint_vector)

			# logging.info(f'Start adding features for {combination}')
			data = AddFeatures(data, combination, self.__features_rolling_window_days)

			# logging.info(f'Start adding target for {combination}')
			data = self.__AddTargetFunc(data, combination, target_col='spread', resulting_target_column='TARGET', target_params=self.__target_params)

			logging.info(f'Finished data creation for {combination}')
		except:
			logging.exception(f'Error adding features for {combination}')
			del data
			gc.collect()
			return None, combination, coint_vector

		gc.collect()
		return data, combination, coint_vector

	def __prepare_all_combination_datas(self, good_combinations: list[tuple[tuple[str, str], PhillipsOuliarisTestResults]], data: pd.DataFrame):
		logging.info(f'Start features and target preparations for {len(good_combinations)} combinations on '
					 f'data set from {data.index[0]} to {data.index[-1]}')

		params = []
		for comb, coint_vector in good_combinations:
			pair1 = comb[0].split('_')[1]
			pair2 = comb[1].split('_')[1]
			comb_columns = [col for col in data.columns if pair1 in col or pair2 in col]
			params.append((data[comb_columns], comb, coint_vector))

		all_results = (Parallel(n_jobs=self.__n_jobs, prefer="processes")
					   (delayed(self.prepare_combination_data)(*p) for p in tqdm(params, total=len(params), desc=f"Train data preparations:")))
		# all_results = parallel(delayed(self.prepare_combination_data)(*p) for p in params)

		# batch_size = multiprocessing.cpu_count()
		# all_results = []
		#
		# for batch_num, i in enumerate(range(0, len(params), batch_size)):
		# 	batch_params = params[i:i + batch_size]
		#
		# 	# results = (Parallel(n_jobs=self.__n_jobs, prefer="processes", backend='multiprocessing')
		# 	# 		   (delayed(self.prepare_combination_data)(*p) for p in
		# 	# 			tqdm(batch_params, total=len(batch_params), desc=f"Batch {batch_num}. Train data preparations:")))
		#
		# 	results = (parallel(n_jobs=self.__n_jobs, prefer="processes", backend='multiprocessing')
		# 			   (delayed(self.prepare_combination_data)(*p) for p in
		# 				tqdm(batch_params, total=len(batch_params), desc=f"Batch {batch_num}. Train data preparations:")))
		#
		# 	results = [tup for tup in results if tup is not None and tup[0] is not None]
		# 	logging.info(f'Got {len(results)} data tuples for batch {batch_num}')
		#
		# 	all_results.extend(results)

		# logging.info('Waiting for loky workers to shutdown')
		# get_reusable_executor().shutdown(wait=True)

		return all_results

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

	def __calc_metrics(self, mtm: np.ndarray, trading_days: float, num_trades: int = None):
		if num_trades is None: num_trades = 0

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
				   'recoveryFactor'       : recoveryFactor,
				   'numTrades'            : num_trades}
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
		signals = []

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
			signals.append(prediction)

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
		stats_df[f'signals'] = signals

		num_trades = np.count_nonzero(np.array(signals))
		metrics = self.__calc_metrics(np.array(combination_mtm_max_capital_based), trading_days, num_trades)
		# mtm_returns_cumprod = (stats_df[f'mtm_returns'] + 1).cumprod()
		# test_metrics_cumprod = self.__calc_metrics(mtm_returns_cumprod.to_numpy(), trading_days)

		del combination_pos, pair_cash_pos, pair_pos, pair_mtm, coef_history, signals

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

		del individual_dfs, iteration_portfolio_df

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

	def __get_val_metrics(self, data_tuples, start_date: datetime, end_train_date: datetime, end_val_date: datetime, end_test_date: datetime):
		val_comb_metrics_tups = []
		for comb_data, combination, coint_vector in data_tuples:
			comb_train = comb_data[(comb_data.index > start_date) & (comb_data.index <= end_train_date)]
			comb_val = comb_data[(comb_data.index > end_train_date) & (comb_data.index <= end_val_date)]
			comb_test = comb_data[(comb_data.index > end_val_date) & (comb_data.index <= end_test_date)]

			preds, model = Train(comb_train, comb_val, combination, self.__ml_val_window_days)
			stats_df, val_metrics = self.__trading_logic(combination, comb_val, preds, coint_vector)
			del stats_df

			val_comb_metrics_tups.append((combination, val_metrics, coint_vector, model, comb_test, comb_val))

		val_comb_metrics_tups.sort(key=lambda x: x[1]['annualized_net_return'], reverse=True)

		return val_comb_metrics_tups

	def __choose_best_combinations(self, val_comb_metrics_tups):
		used_pairs = set()
		combinations_to_trade = []
		for tup in val_comb_metrics_tups:
			combination, val_metrics, coint_vector, model, comb_test, _ = tup
			if combination[0] in used_pairs or combination[1] in used_pairs: continue
			if val_metrics['annualized_net_return'] < self.__min_val_net_return: continue
			if val_metrics['numTrades'] < self.__min_val_num_trades: continue

			combinations_to_trade.append(tup)

		return combinations_to_trade

	def Run(self):
		for start_date, end_train_date, end_val_date, end_test_date in self.__date_bounds:
			all_slice = self.__prices_df[(self.__prices_df.index > start_date) & (self.__prices_df.index <= end_test_date)]
			all_train = self.__prices_df[(self.__prices_df.index > start_date) & (self.__prices_df.index <= end_train_date)]

			good_combinations = SearchForGoodCombinations(all_train,
														  self.__all_possible_combinations,
														  self.__comovement_type,
														  self.__n_jobs,
														  self.__num_good_combs_to_choose)

			data_tuples = self.__prepare_all_combination_datas(good_combinations, all_slice)
			val_comb_metrics_tups = self.__get_val_metrics(data_tuples, start_date, end_train_date, end_val_date, end_test_date)
			combinations_to_trade = self.__choose_best_combinations(val_comb_metrics_tups)
			comb_stats_tups = []
			for combination, val_metrics, coint_vector, model, comb_test, comb_val in combinations_to_trade:
				preds = Predict(comb_test, comb_val, model, combination, self.__use_top_model)
				stats_df, test_metrics = self.__trading_logic(combination, comb_test, preds, coint_vector)
				comb_stats_tups.append((combination, stats_df))

			if comb_stats_tups:
				self.__combine_results(comb_stats_tups)

			del data_tuples, val_comb_metrics_tups, combinations_to_trade, comb_stats_tups
			gc.collect()

		self.__save_all_resulting_metrics()
