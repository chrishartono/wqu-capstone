import gc
import logging
import os
import sys
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

from bottop_prediction import Predict, ResearchTrain, Train, TopModelType
from combinations import SearchForGoodCombinations
from comovement import ComovementType
from feature_engineering import AddFeatures
from spread import AddCointCoefSpread, AddPriceChangeSpread, AddRollingOLSSpread
from target_creation import AddClassificationOLSTarget, AddPeakNeighboursTarget, TargetType
from utils.data_structures import SignalTypes
from utils.helpers import DaysWindowToPeriods, SemiStd


class Backtester:

	def __init__(self,
				 prices_df: pd.DataFrame,
				 train_window_days: int,
				 ml_val_window_days: int,
				 trade_window_days: int,
				 val_window_days: int,
				 features_rolling_windows_days_list: list[int],
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
				 target_params: dict,
				 close_on_no_signal: bool,
				 spread_window: int):

		self.__backtest_id = str(uuid.uuid4())
		logging.info(f'Backtest_id={self.__backtest_id} {train_window_days=} {trade_window_days=} {min_val_net_return=} {min_val_num_trades=} '
					 f'{num_good_combs_to_choose=} {target_params=} {close_on_no_signal=}')

		self.__prices_df = prices_df
		self.__train_window_days = train_window_days
		self.__trade_window_days = trade_window_days
		self.__ml_val_window_days = ml_val_window_days
		self.__features_rolling_windows_days_list = features_rolling_windows_days_list
		self.__all_possible_combinations = all_possible_combinations
		self.__comovement_type = comovement_detection_type
		self.__date_bounds = self.__make_date_bounds(prices_df, train_window_days, val_window_days, trade_window_days)
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

		self.__close_on_no_signal = close_on_no_signal
		self.__spread_window = spread_window

		self.__annualized_multiplier = np.sqrt(24 * 365)

		self.__portfolio_df = None
		self.__stats_by_comb = {}

		self.__use_top_model = use_top_model

		self.__main_path, self.__aggregated_path = self.__create_main_paths()

	def __create_main_paths(self):
		now = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
		main_path = f'results/{now}_{self.__backtest_id}'
		aggregated_path = f'{main_path}/aggregated'

		os.makedirs(main_path, exist_ok=True)
		os.makedirs(aggregated_path, exist_ok=True)

		return main_path, aggregated_path

	def __make_date_bounds(self, prices_df: pd.DataFrame, train_window_days: int, val_window_days: int, trade_window_days: int):
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

		train_window = timedelta(days=train_window_days)
		val_window = timedelta(days=val_window_days)
		test_window = timedelta(days=trade_window_days)

		date_bounds = []
		# Iterate until there is not enough data to have at least val_test_days for train and val_test_days for val and test
		while current_bound_date + train_window + val_window <= last_date:

			if current_bound_date + train_window + val_window + test_window <= last_date:  # The whole wf_window fits before last_date
				date_bounds.append((current_bound_date, current_bound_date + train_window, current_bound_date + train_window + val_window,
									current_bound_date + train_window + val_window + test_window))
			else:  # Only 2 val_test_days windows fit before last_date
				date_bounds.append((
						current_bound_date, current_bound_date + train_window, current_bound_date + train_window + val_window, last_date))
				break

			current_bound_date = current_bound_date + test_window

		return date_bounds

	def prepare_combination_data(self,
								 data: pd.DataFrame,
								 combination: tuple[str, str],
								 coint_vector: PhillipsOuliarisTestResults,
								 end_train_date: datetime):
		coefs_vector = None
		try:
			# logging.info(f'Start adding spread for {combination}')
			# data = AddCointCoefSpread(data, combination, coint_vector)
			# data = AddPriceChangeSpread(data, combination)
			data, coefs_vector = AddRollingOLSSpread(data, combination, self.__spread_window)

			# logging.info(f'Start adding features for {combination}')
			data, categorical_features = AddFeatures(data,
													 combination,
													 self.__features_rolling_windows_days_list,
													 end_train_date)

			data = self.__AddTargetFunc(data, combination, target_col='spread', resulting_target_column='TARGET', target_params=self.__target_params)

			# logging.info(f'Start adding target for {combination}')

			if coefs_vector is None:
				c0 = combination[0]
				c1 = combination[1]
				coefs_vector = {c0: np.full(len(data), fill_value=coint_vector[c0]), c1: np.full(len(data), fill_value=coint_vector[c1])}

			coefs_df = pd.DataFrame(coefs_vector, index=data.index)

			logging.info(f'Finished data creation for {combination}')
		except:
			logging.exception(f'Error adding features for {combination}')
			del data
			gc.collect()
			return None, combination, None, None

		gc.collect()
		return data, combination, coefs_df, categorical_features

	def __prepare_all_combination_datas(self, good_combinations: list[tuple[tuple[str, str], PhillipsOuliarisTestResults]],
										data: pd.DataFrame,
										end_train_date: datetime):
		logging.info(f'Start features and target preparations for {len(good_combinations)} combinations on '
					 f'data set from {data.index[0]} to {data.index[-1]}')

		params = []
		for comb, coint_vector in good_combinations:
			pair1 = comb[0].split('_')[1]
			pair2 = comb[1].split('_')[1]
			comb_columns = [col for col in data.columns if pair1 in col or pair2 in col]
			params.append((data[comb_columns], comb, coint_vector, end_train_date))

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

	def __update_stats(self,
					   prices,
					   combination_pos,
					   pair_cash_pos,
					   pair_pos,
					   pair_mtm,
					   fees_history,
					   coef_history,
					   combination_exposure_trades,
					   last_pair_cash_pos,
					   last_pair_pos,
					   last_fees,
					   coef,
					   i):
		combination_pos.append(combination_exposure_trades)
		pair_cash_pos[0].append(last_pair_cash_pos[0])
		pair_cash_pos[1].append(last_pair_cash_pos[1])
		pair_pos[0].append(last_pair_pos[0])
		pair_pos[1].append(last_pair_pos[1])
		pair_mtm[0].append(last_pair_cash_pos[0] + last_pair_pos[0] * prices[0][i])
		pair_mtm[1].append(last_pair_cash_pos[1] + last_pair_pos[1] * prices[1][i])
		fees_history[0].append(last_fees[0])
		fees_history[1].append(last_fees[1])
		coef_history[0].append(coef[0])
		coef_history[1].append(coef[1])

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

	def __trading_logic(self, combination: tuple[str, str], test: pd.DataFrame, preds: np.ndarray, coefs_df: pd.DataFrame):
		pair0 = combination[0]
		pair1 = combination[1]
		coef_orig = [coefs_df[pair0].to_numpy(), coefs_df[pair1].to_numpy()]
		prices = [test[pair0].to_numpy(), test[pair1].to_numpy()]
		spread_orig = prices[0] * coef_orig[0] + prices[1] * coef_orig[1]

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
		fees_history = [[], []]
		signals = []

		# Points where we actually make trades
		buy_points = np.full(len(test), np.nan)
		sell_points = np.full(len(test), np.nan)

		# Points where we get signals to buy or sell
		buy_signals = np.full(len(test), np.nan)
		sell_signals = np.full(len(test), np.nan)

		# Max number of trades we can open according to backtest settings: trade_limit and combination_limit
		max_exposure = int(self.__combination_limit / self.__trade_limit)

		# Current strategy exposure in number of open trades
		combination_exposure_trades = 0

		# Latest cumulative cash position for each pair. If we buy pair, we spend cash, so cash position is negative. And vice versa.
		last_pair_cash_pos = [0, 0]

		# Latest cumulative pair position in trades. Buy once, get position=1.
		last_pair_pos = [0, 0]

		num_trades = 0

		for i, prediction in enumerate(preds):
			prediction = prediction[0]
			# coef = [coef_orig[0] * coef_adjustment, coef_orig[1] * coef_adjustment]
			coef = [coef_orig[0][i], coef_orig[1][i]]

			last_fees = [0, 0]
			signals.append(prediction)

			if prediction == SignalTypes.BUY.value:
				buy_signals[i] = spread_orig[i]

				if combination_exposure_trades + 1 <= max_exposure:
					# We BUY spread. Increase strategy exposure by 1 trade
					combination_exposure_trades += 1
					buy_points[i] = spread_orig[i]
					# If we BUY spread, it means that we use coefs with signs as they were given by cointegration.
					# Say we had cointegration coefs [0.8, -3.8]. Buying spread means buying 0.8 pair0 and selling 3.8 pair1.
					# But cash flow has opposite sign. So we have spent cash to buy 0.8 pair0 and earned  after selling 3.8 pair1.
					# To represent this we subtract trade volume from last cash_pos. If coef > 0, it means we should buy, then we subtract trade volume.
					# If coef < 0, it means we should sell, then minus * minus gives plus => we add trade volume.
					# And as this is a cumulative cash_pos, we add it to the previous one.
					last_pair_cash_pos = [last_pair_cash_pos[0] - prices[0][i] * coef[0], last_pair_cash_pos[1] - prices[1][i] * coef[1]]
					# Positions counted as number of trades. Add to the previous.
					last_pair_pos = [last_pair_pos[0] + coef[0], last_pair_pos[1] + coef[1]]
					last_fees = [abs(prices[0][i] * coef[0] * self.__fees), abs(prices[1][i] * coef[1] * self.__fees)]

			elif prediction == SignalTypes.SELL.value:
				sell_signals[i] = spread_orig[i]

				if combination_exposure_trades - 1 >= -max_exposure:
					combination_exposure_trades -= 1
					sell_points[i] = spread_orig[i]
					# Opposite here. Flip the signs.
					last_pair_cash_pos = [last_pair_cash_pos[0] + prices[0][i] * coef[0], last_pair_cash_pos[1] + prices[1][i] * coef[1]]
					last_pair_pos = [last_pair_pos[0] - coef[0], last_pair_pos[1] - coef[1]]
					last_fees = [abs(prices[0][i] * coef[0] * self.__fees), abs(prices[1][i] * coef[1] * self.__fees)]

			elif self.__close_on_no_signal and combination_exposure_trades != 0:  # No signal, close position if it is open
				if combination_exposure_trades > 0: sell_points[i] = spread_orig[i]
				else: buy_points[i] = spread_orig[i]

				last_pair_cash_pos = [last_pair_cash_pos[0] + prices[0][i] * last_pair_pos[0], last_pair_cash_pos[1] + prices[1][i] * last_pair_pos[1]]
				last_fees = [abs(prices[0][i] * last_pair_pos[0] * self.__fees), abs(prices[1][i] * last_pair_pos[1] * self.__fees)]
				last_pair_pos = [0, 0]
				combination_exposure_trades = 0

			# Adding fees directly to cash_flow so they can accumulate
			last_pair_cash_pos = [last_pair_cash_pos[0] - last_fees[0], last_pair_cash_pos[1] - last_fees[1]]


			if (last_fees[0] > 0 or last_fees[1] > 0): num_trades += 1

			# Add all last values to the lists of running statistics.
			self.__update_stats(prices,
								combination_pos,
								pair_cash_pos,
								pair_pos,
								pair_mtm,
								fees_history,
								coef_history,
								combination_exposure_trades,
								last_pair_cash_pos,
								last_pair_pos,
								last_fees,
								coef,
								i)

		trading_days = (test.index[-1] - test.index[0]).days
		combination_mtm_0based = [t[0] + t[1] for t in zip(*pair_mtm)]
		coef_history_arrays = [np.array(coef_history[0]), np.array(coef_history[1])]
		capital_usage = abs(prices[0] * coef_history_arrays[0]) + abs(prices[1] * coef_history_arrays[1])
		max_capital_usage = max(capital_usage)

		# If we made no trades, max_capital_usage=0. But combination_mtm_max_capital_based list will also contain only zeros then.
		# And in the end we will have issues with calculating metrics
		max_capital_usage = max(max_capital_usage, 0.01)
		combination_mtm_max_capital_based = np.array([mtm + max_capital_usage for mtm in combination_mtm_0based])
		mtm_for_portfolio = combination_mtm_max_capital_based.copy()

		# Here we zero out all mark-to-market values after it reached zero or crossed it to negative region. Strategy has lost all its capital.
		neg_mask = mtm_for_portfolio < 0
		if np.any(neg_mask):
			first_neg = np.argmax(neg_mask)  # index of first True in neg_mask
			mtm_for_portfolio[first_neg:] = 0

		metrics = self.__calc_metrics(mtm_for_portfolio, trading_days, num_trades)

		stats_df = pd.DataFrame(buy_signals, index=test.index, columns=['buy_signals'])
		stats_df['sell_signals'] = sell_signals
		stats_df['buy_points'] = buy_points
		stats_df['sell_points'] = sell_points

		stats_df['mtm'] = mtm_for_portfolio
		stats_df[f'mtm_non_capped'] = combination_mtm_max_capital_based
		stats_df[f'mtm_returns'] = stats_df['mtm'].pct_change()
		stats_df[f'mtm_returns'].replace([np.inf, -np.inf], np.nan, inplace=True)
		stats_df[f'mtm_returns'].fillna(0, inplace=True)

		stats_df[pair0] = prices[0]
		stats_df[f'coef_{pair0}'] = coef_history[0]
		stats_df[f'cash_pos_{pair0}'] = pair_cash_pos[0]
		stats_df[f'pos_qty_{pair0}'] = pair_pos[0]
		stats_df[f'mtm_{pair0}'] = pair_mtm[0]
		stats_df[f'fees_{pair0}'] = fees_history[0]

		stats_df[pair1] = prices[1]
		stats_df[f'coef_{pair1}'] = coef_history[1]
		stats_df[f'cash_pos_{pair1}'] = pair_cash_pos[1]
		stats_df[f'pos_qty_{pair1}'] = pair_pos[1]
		stats_df[f'mtm_{pair1}'] = pair_mtm[1]
		stats_df[f'fees_{pair1}'] = fees_history[1]

		stats_df['spread'] = spread_orig
		stats_df[f'pos'] = combination_pos
		stats_df[f'signals'] = signals

		del combination_pos, pair_cash_pos, pair_pos, pair_mtm, coef_history, signals

		return stats_df, metrics

	def __combine_results(self, comb_stats_tups: list[tuple[tuple[str, str], pandas.DataFrame, dict]]):

		individual_dfs = []
		columns = []
		for combination, stats_df, _ in comb_stats_tups:
			pair0 = combination[0]
			pair1 = combination[1]

			columns.append(f'{pair0}_{pair1}_mtm_returns')

			if combination not in self.__stats_by_comb:
				self.__stats_by_comb[combination] = stats_df
			else:
				old_stats_df = self.__stats_by_comb[combination]
				self.__stats_by_comb[combination] = pd.concat([old_stats_df, stats_df], axis=0)

			individual_dfs.append(stats_df['mtm_returns'])

		iteration_portfolio_df = pd.concat(individual_dfs, axis=1)
		iteration_portfolio_df.columns = columns

		# We assume equal capital allocations across all combinations
		iteration_portfolio_df['portfolio_returns'] = iteration_portfolio_df.mean(axis=1)
		iteration_portfolio_df['active_combinations'] = len(individual_dfs)

		if self.__portfolio_df is None:
			self.__portfolio_df = iteration_portfolio_df[['portfolio_returns', 'active_combinations']]
		else:
			self.__portfolio_df = pd.concat([self.__portfolio_df, iteration_portfolio_df[['portfolio_returns', 'active_combinations']]], axis=0)

		del individual_dfs, iteration_portfolio_df

	def __save_plot(self, pair0: str, pair1: str, stats_df: pd.DataFrame, metrics: dict, plot_path: str, save_file_name: str):
		nrows = 3 if 'spread' in stats_df.columns else 1

		if 'pos' in stats_df.columns: nrows += 1
		if 'active_combinations' in stats_df.columns: nrows += 1

		fig, ax = plt.subplots(nrows, ncols=1, figsize=(35, 25))
		row_ax = 0

		if 'spread' in stats_df.columns:
			twinx1 = ax[row_ax].twinx()
			stats_df[pair0].plot(ax=ax[row_ax], color='blue', label=pair0)
			stats_df[pair1].plot(ax=twinx1, color='orange', label=pair1)
			lines, labels = ax[row_ax].get_legend_handles_labels()
			lines2, labels2 = twinx1.get_legend_handles_labels()
			ax[row_ax].legend(lines + lines2, labels + labels2, loc=0)

			row_ax += 1
			ax[row_ax].plot(stats_df.index, stats_df['spread'], color='grey')
			ax[row_ax].plot(stats_df.index, stats_df['buy_points'], 'g^', markersize=5)
			ax[row_ax].plot(stats_df.index, stats_df['sell_points'], 'rv', markersize=5)
			ax[row_ax].set_title(f'{pair0}-{pair1}. Sharpe={metrics["sharpe"]:.2f} '
							  f'Net_return={metrics["annualized_net_return"]:.2f} '
							  f'MaxDD_pct={metrics["maxDD"]:.2f} '
							  f'Num_trades={metrics["numTrades"]:.0f}')

			twinx2 = ax[row_ax].twinx()
			twinx2.plot(stats_df.index, stats_df['cumprod_mtm_returns'], color='magenta')

			row_ax += 1
			ax[row_ax].plot(stats_df.index, stats_df['spread'], color='grey')
			ax[row_ax].plot(stats_df.index, stats_df['buy_signals'], 'g^', markersize=5)
			ax[row_ax].plot(stats_df.index, stats_df['sell_signals'], 'rv', markersize=5)
			ax[row_ax].set_title(f'All signals')
		elif 'cumprod_mtm_returns' in stats_df.columns:
			ax[row_ax].plot(stats_df.index, stats_df['cumprod_mtm_returns'], color='magenta')

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

		plt.tight_layout()
		fig.savefig(os.path.join(plot_path, f'{save_file_name}.png'))
		plt.close()

	def __save_iteration_results(self,
								 iteration: int,
								 start_date: datetime,
								 end_date: datetime,
								 comb_stats_tups: list[tuple[tuple[str, str], pandas.DataFrame, dict]]):
		start_date_str = start_date.strftime('%Y-%m-%d_%H-%M-%S')
		end_date_str = end_date.strftime('%Y-%m-%d_%H-%M-%S')
		iteration_dir = f'{iteration}_({start_date_str})_({end_date_str})'

		iteration_path = f'{self.__main_path}/{iteration_dir}'
		iteration_plot_path = f'{self.__main_path}/{iteration_dir}/plots'
		iteration_equity_path = f'{self.__main_path}/{iteration_dir}/equity'

		os.makedirs(iteration_path, exist_ok=True)
		os.makedirs(iteration_plot_path, exist_ok=True)
		os.makedirs(iteration_equity_path, exist_ok=True)

		individual_dfs = []
		columns = []
		for combination, stats_df, metrics in comb_stats_tups:
			pair0 = combination[0]
			pair1 = combination[1]

			columns.append(f'{pair0}_{pair1}_mtm_returns')

			save_file_name = f'{pair0}_{pair1}_{iteration}'

			stats_df['cumprod_mtm_returns'] = (stats_df[f'mtm_returns'] + 1).cumprod()
			stats_df.to_csv(os.path.join(iteration_equity_path, f'{save_file_name}.csv'), index=True, index_label='date')
			self.__save_plot(pair0, pair1, stats_df, metrics, iteration_plot_path, save_file_name)

			individual_dfs.append(stats_df['mtm_returns'])

		iteration_portfolio_df = pd.concat(individual_dfs, axis=1)
		iteration_portfolio_df.columns = columns

		# We assume equal capital allocations across all combinations
		iteration_portfolio_df['portfolio_returns'] = iteration_portfolio_df.mean(axis=1)
		iteration_portfolio_df['active_combinations'] = len(individual_dfs)

		if iteration_portfolio_df is not None and len(iteration_portfolio_df) > 0:
			iteration_portfolio_df['cumprod_mtm_returns'] = (iteration_portfolio_df['portfolio_returns'] + 1).cumprod()
			trading_days = (iteration_portfolio_df.index[-1] - iteration_portfolio_df.index[0]).days
			portfolio_metrics = self.__calc_metrics(iteration_portfolio_df['cumprod_mtm_returns'].to_numpy(), trading_days)

			iteration_portfolio_df.to_csv(os.path.join(iteration_equity_path, f'portfolio_equity_{iteration}.csv'), index=True, index_label='date')
			self.__save_plot('portfolio', 'portfolio', iteration_portfolio_df, portfolio_metrics, iteration_plot_path, f'portfolio_plot_{iteration}')

		del individual_dfs, iteration_portfolio_df

	def __save_all_results(self):
		aggregated_plot_path = f'{self.__aggregated_path}/plots'
		aggregated_equity_path = f'{self.__aggregated_path}/equity'
		os.makedirs(aggregated_plot_path, exist_ok=True)
		os.makedirs(aggregated_equity_path, exist_ok=True)

		for combination, stats_df in self.__stats_by_comb.items():
			pair0 = combination[0]
			pair1 = combination[1]

			stats_df['cumprod_mtm_returns'] = (stats_df[f'mtm_returns'] + 1).cumprod()
			trading_days = (stats_df.index[-1] - stats_df.index[0]).days
			num_trades = stats_df['buy_points'].count() + stats_df['sell_points'].count()
			metrics = self.__calc_metrics(stats_df['cumprod_mtm_returns'].to_numpy(), trading_days, num_trades)

			save_file_name = f'{pair0}_{pair1}'

			stats_df.to_csv(os.path.join(aggregated_equity_path, f'{save_file_name}.csv'), index=True, index_label='date')

			self.__save_plot(pair0, pair1, stats_df, metrics, aggregated_plot_path, save_file_name)

		if self.__portfolio_df is not None and len(self.__portfolio_df) > 0:
			self.__portfolio_df['cumprod_mtm_returns'] = (self.__portfolio_df['portfolio_returns'] + 1).cumprod()
			trading_days = (self.__portfolio_df.index[-1] - self.__portfolio_df.index[0]).days
			portfolio_metrics = self.__calc_metrics(self.__portfolio_df['cumprod_mtm_returns'].to_numpy(), trading_days)

			self.__portfolio_df.to_csv(os.path.join(aggregated_equity_path, f'portfolio_equity.csv'), index=True, index_label='date')
			self.__save_plot('portfolio', 'portfolio', self.__portfolio_df, portfolio_metrics, aggregated_plot_path, 'portfolio_plot')

	def __get_val_metrics(self, data_tuples, start_date: datetime, end_train_date: datetime, end_val_date: datetime, end_test_date: datetime):
		val_comb_metrics_tups = []
		for comb_data, combination, coefs_df, categorical_features in data_tuples:
			comb_train = comb_data[(comb_data.index > start_date) & (comb_data.index <= end_train_date)]
			comb_val = comb_data[(comb_data.index > end_train_date) & (comb_data.index <= end_val_date)]
			comb_test = comb_data[(comb_data.index > end_val_date) & (comb_data.index <= end_test_date)]

			coefs_df_val = coefs_df[(coefs_df.index > end_train_date) & (coefs_df.index <= end_val_date)]
			coefs_df_test = coefs_df[(coefs_df.index > end_val_date) & (coefs_df.index <= end_test_date)]

			preds, model = Train(comb_train, comb_val, combination, self.__ml_val_window_days, categorical_features)
			stats_df, val_metrics = self.__trading_logic(combination, comb_val, preds, coefs_df_val)
			del stats_df

			val_comb_metrics_tups.append((combination, val_metrics, coefs_df_test, model, comb_test, comb_val))

		val_comb_metrics_tups.sort(key=lambda x: x[1]['annualized_net_return'], reverse=True)

		return val_comb_metrics_tups

	def __choose_best_combinations(self, val_comb_metrics_tups):
		used_pairs = set()
		combinations_to_trade = []
		for tup in val_comb_metrics_tups:
			combination, val_metrics, coefs_df, model, comb_test, _ = tup
			if combination[0] in used_pairs or combination[1] in used_pairs: continue
			if val_metrics['annualized_net_return'] < self.__min_val_net_return: continue
			if val_metrics['numTrades'] < self.__min_val_num_trades: continue

			combinations_to_trade.append(tup)
			used_pairs.add(combination[0])
			used_pairs.add(combination[1])

		return combinations_to_trade

	def Run(self):
		logging.info(f'Will run {len(self.__date_bounds)} iterations')
		for iteration, (start_date, end_train_date, end_val_date, end_test_date) in enumerate(self.__date_bounds):
			logging.info(f'Start iteration {iteration}')
			#
			# start_date = '2023-10-18 00:00:00'
			# end_train_date = '2024-08-12 00:00:00'
			# end_val_date = '2024-09-11 00:00:00'
			# end_test_date = '2024-10-11 23:00:00 '

			all_slice = self.__prices_df[(self.__prices_df.index > start_date) & (self.__prices_df.index <= end_test_date)]
			all_train = self.__prices_df[(self.__prices_df.index > start_date) & (self.__prices_df.index <= end_train_date)]

			good_combinations = SearchForGoodCombinations(all_train,
														  self.__all_possible_combinations,
														  self.__comovement_type,
														  self.__n_jobs,
														  self.__num_good_combs_to_choose)

			# sys.exit(0)

			data_tuples = self.__prepare_all_combination_datas(good_combinations, all_slice, end_train_date)
			val_comb_metrics_tups = self.__get_val_metrics(data_tuples, start_date, end_train_date, end_val_date, end_test_date)
			combinations_to_trade = self.__choose_best_combinations(val_comb_metrics_tups)
			comb_stats_tups = []
			for combination, val_metrics, coefs_df_test, model, comb_test, comb_val in combinations_to_trade:
				preds = Predict(comb_test, comb_val, model, combination, self.__use_top_model)
				stats_df, test_metrics = self.__trading_logic(combination, comb_test, preds, coefs_df_test)
				comb_stats_tups.append((combination, stats_df, test_metrics))

			if comb_stats_tups:
				self.__combine_results(comb_stats_tups)
				self.__save_iteration_results(iteration, start_date, end_test_date, comb_stats_tups)

			del data_tuples, val_comb_metrics_tups, combinations_to_trade, comb_stats_tups
			gc.collect()

		self.__save_all_results()

	def __plot_metrics_hist(self, all_metrics: dict[str, list[float]], filename: str):

		def plot_group(sorted_items, group_name: str, ax, row: int):
			for i, (label, values) in enumerate(sorted_items):
				# Freedman–Diaconis rule
				q25, q75 = np.percentile(values, [25, 75])
				bin_width = 2 * (q75 - q25) * len(values) ** (-1 / 3)
				bins = round((max(values) - min(values)) / bin_width)
				ax[row, i].hist(values, bins=bins)

				mean = np.mean(values)
				median = np.median(values)
				std = np.std(values)
				ax[row, i].set_title(f'{label} Mean={mean:.3f}. Median={median:.3f}. Std={std:.3f}')

		auc_items = [(k, v) for k, v in all_metrics.items() if 'auc' in k]
		auc_items.sort(key=lambda x: x[0])

		f1_default_items = [(k, v) for k, v in all_metrics.items() if 'f1_default' in k]
		f1_default_items.sort(key=lambda x: x[0])

		f1_tuned_items = [(k, v) for k, v in all_metrics.items() if 'f1_tuned' in k]
		f1_tuned_items.sort(key=lambda x: x[0])

		fig, axes = plt.subplots(3, ncols=len(auc_items), figsize=(35, 20))

		plot_group(auc_items, group_name='AUC', ax=axes, row=0)
		plot_group(f1_default_items, group_name='F1 Default Threshold', ax=axes, row=1)
		plot_group(f1_tuned_items, group_name='F1 Tuned Threshold', ax=axes, row=2)

		fig.savefig(f'{self.__main_path}/metrics_distr_{filename}_{self.__backtest_id}.png')
		plt.close()

	def __make_aggregated_ml_stats(self, all_metrics: dict[str, list[float]]):
		def make_group_aggr(metric: str, items):
			aggr_values = []
			for label, values in items:
				label_class = label.split('_')[-1]
				median = np.median(values)
				std = np.std(values)
				aggr_values.append({'class': label_class, 'metric': metric, 'median': median, 'std': std})

			return aggr_values

		all_aggr_values = []
		auc_items = [(k, v) for k, v in all_metrics.items() if 'auc' in k]
		auc_aggr = make_group_aggr('auc', auc_items)
		all_aggr_values.extend(auc_aggr)

		f1_default_items = [(k, v) for k, v in all_metrics.items() if 'f1_default' in k]
		f1_default_aggr = make_group_aggr('f1_default', f1_default_items)
		all_aggr_values.extend(f1_default_aggr)

		f1_tuned_items = [(k, v) for k, v in all_metrics.items() if 'f1_tuned' in k]
		f1_tuned_aggr = make_group_aggr('f1_tuned', f1_tuned_items)
		all_aggr_values.extend(f1_tuned_aggr)

		return all_aggr_values

	def MLPredictionQualityTest(self, desired_num_samples: int, filename: str):
		sample_size = (self.__train_window_days + self.__ml_val_window_days + self.__trade_window_days) * 24
		possible_num_samples = int(len(self.__prices_df) / sample_size)

		all_slices = np.array_split(self.__prices_df, possible_num_samples)

		if possible_num_samples >= desired_num_samples + 2:
			indices = np.round(np.linspace(1, len(all_slices) - 2, desired_num_samples)).astype(int)
		else:
			indices = np.round(np.linspace(0, len(all_slices) - 1, possible_num_samples)).astype(int)
		selected_slices = [all_slices[i] for i in indices]

		all_metrics = dict()
		for i, df_slice in enumerate(selected_slices):
			logging.info(f'Starting iteration {i} out of {len(selected_slices)}')
			end_train_date = df_slice.index[0] + timedelta(days=self.__train_window_days)
			end_val_date = end_train_date + timedelta(days=self.__ml_val_window_days)

			train = df_slice[df_slice.index <= end_train_date]

			good_combinations = SearchForGoodCombinations(train,
														  self.__all_possible_combinations,
														  self.__comovement_type,
														  self.__n_jobs,
														  self.__num_good_combs_to_choose)
			if not good_combinations: continue

			data_tuples = self.__prepare_all_combination_datas(good_combinations, df_slice, end_train_date)
			if not data_tuples: continue

			for j, (comb_data, combination, coefs_df, categorical_features) in enumerate(data_tuples):
				logging.info(f'Starting combination {j} {combination} out of {len(data_tuples)}')
				comb_train = comb_data[comb_data.index <= end_train_date]
				comb_val = comb_data[(comb_data.index > end_train_date) & (comb_data.index <= end_val_date)]
				comb_test = comb_data[comb_data.index > end_val_date]

				try:
					metrics = ResearchTrain(comb_train, comb_val, comb_test, combination, categorical_features)
				except Exception as e:
					logging.error(f'{combination} train failed with exception: {e}')
					continue

				for label, metric_value in metrics.items():
					if label not in all_metrics: all_metrics[label] = []
					all_metrics[label].append(metric_value)

		self.__plot_metrics_hist(all_metrics, filename)
		all_aggr_values = self.__make_aggregated_ml_stats(all_metrics)

		return all_aggr_values
