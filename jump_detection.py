import copy
import logging
from datetime import timedelta

import numpy as np
import scipy.special as sp
from matplotlib import pyplot as plt
from scipy import stats
import bottleneck as bn
# from matplotlib import pyplot as plt
from sortedcontainers import SortedList

import warnings

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	import pandas as pd


def detect_jump_inc_alternative(returns: np.ndarray,
								jumps: np.ndarray,
								prev_vol: float,
								ret_sorted: SortedList,
								bandwidth: int,
								up_jumps: SortedList,
								down_jumps: SortedList,
								prob_cut_off: float = 0.05,
								debug: bool = False):
	jumps = np.append(jumps, False)

	last_vol = theoretical_vol_inc(returns, bandwidth, jumps)
	last_vol = prev_vol if last_vol is None else last_vol

	last_return_adjusted = returns[-1] / last_vol
	# if debug:
	# 	last_return_adjusted = np.round(last_return_adjusted, decimals=8)

	is_jump, ret_sorted, up_jumps, down_jumps = get_jump_probabilities_inc(last_return_adjusted, ret_sorted, up_jumps, down_jumps, prob_cut_off)

	if abs(returns[-1]) < last_vol: is_jump = False
	jumps[-1] = is_jump

	return last_vol, jumps, ret_sorted, up_jumps, down_jumps


def detect_jumps(returns: np.ndarray, bandwidth: int, prob_cut_off: float = 0.05, max_iter: int = 100, debug: bool = False):
	jumps = np.zeros(len(returns)).astype(bool)
	ret_sorted = None

	for iteration in range(max_iter):
		jumps_old = copy.deepcopy(jumps)
		vols = theoretical_vols(returns, bandwidth, jumps)
		returns_adjusted = returns / vols

		# if debug:
		# 	returns_adjusted = np.round(returns_adjusted, decimals=8)

		jumps, ret_sorted, up_jumps_list, down_jumps_list = get_jump_probabilities(returns_adjusted, jumps, prob_cut_off)

		# Here we neglect jumps lower than volatility estimation as their size is inside the Gaussian region
		jumps[np.abs(returns) < vols] = False

		# No changes in jumps since last iteration. Converged. Can exit.
		if np.sum(jumps_old == jumps) >= len(jumps): break

	# If we exited with break, then vols were already calculated previously with this set of jumps.
	# No need to recalculate.
	if iteration >= max_iter - 1:
		vols = theoretical_vols(returns, bandwidth, jumps)

	up_jumps = SortedList(up_jumps_list)
	down_jumps = SortedList(down_jumps_list)

	return vols, jumps, ret_sorted, up_jumps, down_jumps


# def detect_jump_inc(returns: np.ndarray, jumps: np.ndarray, vols: np.ndarray, bandwidth: int, prob_cut_off: float = 0.05, max_iter: int = 100):
# 	jumps = np.append(jumps, False)
#
# 	for iteration in range(max_iter):
# 		jumps_old = copy.deepcopy(jumps)
#
# 		# Most of the time last return won't be a jump so there is no need to recalculate all volatilities
# 		if iteration == 0:
# 			last_vol = theoretical_vol_inc(returns, bandwidth, jumps)
# 			last_vol = vols[-1] if last_vol is None else last_vol
#
# 			vols = np.append(vols, last_vol)
#
# 			returns_adjusted = returns / vols
# 		# If it is jump we go for next iterations, we have to recalculate all from scratch
# 		else:
# 			vols = theoretical_vols(returns, bandwidth, jumps)
# 			returns_adjusted = returns / vols
#
# 		jumps, ret_sorted, num_gauss_up_returns, num_gauss_down_returns, num_no_jumps = get_jump_probabilities(returns_adjusted, jumps, prob_cut_off)
#
# 		# Here we neglect jumps lower than volatility estimation as their size is inside the Gaussian region
# 		jumps[np.abs(returns) < vols] = False
#
# 		# No changes in jumps since last iteration. Converged. Can exit.
# 		if np.sum(jumps_old == jumps) >= len(jumps): break
#
# 	# If we exited with break, then vols were already calculated previously with this set of jumps.
# 	# No need to recalculate.
# 	if iteration >= max_iter - 1:
# 		vols = theoretical_vols(returns, bandwidth, jumps)
#
# 	return vols, jumps, returns_adjusted


def get_jump_probabilities(returns_adjusted: np.ndarray, jumps: np.ndarray[bool], prob_cut_off: float):
	ret_sorted_idx = np.argsort(returns_adjusted)
	ret_sorted = np.sort(returns_adjusted)

	N = len(returns_adjusted)
	num_no_jumps = len(returns_adjusted)
	num_gauss_down_returns = 1
	num_gauss_up_returns = 1

	up_jumps = []
	down_jumps = []

	# Here we neglect the center of the ret_ren array if the size is odd
	for k in np.arange(1, int(len(returns_adjusted) * 0.5) + 1):
		jumps, num_gauss_down_returns, num_no_jumps = get_kSmallestCDF_single_side(jumps,
																				   ret_sorted[k - 1],
																				   ret_sorted_idx[k - 1],
																				   num_gauss_down_returns,
																				   num_no_jumps,
																				   prob_cut_off)
		if jumps[ret_sorted_idx[k - 1]]:
			down_jumps.append(ret_sorted[k - 1])

		jumps, num_gauss_up_returns, num_no_jumps = get_kSmallestCDF_single_side(jumps,
																				 -ret_sorted[N - k],
																				 ret_sorted_idx[N - k],
																				 num_gauss_up_returns,
																				 num_no_jumps,
																				 prob_cut_off)
		if jumps[ret_sorted_idx[N - k]]:
			up_jumps.append(ret_sorted[N - k])

	return jumps, ret_sorted, up_jumps, down_jumps


def get_jump_probabilities_inc(last_return: float, ret_sorted: SortedList, up_jumps: SortedList, down_jumps: SortedList, prob_cut_off: float):
	insert_idx = ret_sorted.bisect_left(last_return)

	return_rank = insert_idx if insert_idx <= len(ret_sorted) * 0.5 else len(ret_sorted) - (insert_idx - 1)

	down_jump_rank = down_jumps.bisect_left(-abs(last_return))
	num_gauss_down_returns = max(return_rank - (down_jump_rank - 1), 1)

	insert_up_jump_idx = up_jumps.bisect_left(abs(last_return))
	up_jump_rank = len(up_jumps) - insert_up_jump_idx
	num_gauss_up_returns = max(return_rank - (up_jump_rank - 1), 1)

	num_no_jumps = len(ret_sorted) - down_jump_rank - up_jump_rank + 2

	if insert_idx <= len(ret_sorted) * 0.5:
		is_jump, num_gauss_down_returns, num_no_jumps = get_kSmallestCDF_single_side_inc(last_return, num_gauss_down_returns, num_no_jumps, prob_cut_off)
		if is_jump:
			down_jumps.add(last_return)
	else:
		is_jump, num_gauss_up_returns, num_no_jumps = get_kSmallestCDF_single_side_inc(-last_return, num_gauss_up_returns, num_no_jumps, prob_cut_off)
		if is_jump:
			up_jumps.add(last_return)

	ret_sorted.add(last_return)

	return is_jump, ret_sorted, up_jumps, down_jumps

def theoretical_vols(returns: np.ndarray, bandwidth: int, jumps: np.ndarray):
	no_jumps = 1. - jumps
	vols = np.full(len(returns), np.nan)

	for i in np.arange(bandwidth - 1, len(returns)):
		returns_without_jumps = returns[i + 1 - bandwidth:i + 1] * no_jumps[i + 1 - bandwidth:i + 1]
		num_no_jumps = np.sum(no_jumps[i + 1 - bandwidth:i + 1])

		# No need to calculate anything more if num_no_jumps=0. This vol value will stay NaN. Will ffill in the end.
		if num_no_jumps == 0: continue

		vols[i] = np.sum(returns_without_jumps ** 2.) / num_no_jumps
		vols[i] = np.sqrt(vols[i])

	vols[:bandwidth] = vols[bandwidth - 1]

	# ffill
	vols = bn.push(vols)

	# bfill
	vols = bn.push(vols[::-1])[::-1]
	return vols


def theoretical_vol_inc(returns: np.ndarray, bandwidth: int, jumps: np.ndarray):
	no_jumps = 1. - jumps

	returns_without_jumps = returns[-bandwidth:] * no_jumps[-bandwidth:]

	num_no_jump = np.sum(no_jumps[-bandwidth:])

	if num_no_jump == 0:
		return None

	vol = np.sum(returns_without_jumps ** 2.) / num_no_jump
	vol = np.sqrt(vol)
	return vol


def get_kSmallestCDF_single_side(jumps: np.ndarray[bool], cur_return: float, return_idx: int, num_returns_gauss: int, num_no_jumps: int, prob_cut_off: float):
	if jumps[return_idx]:
		num_no_jumps -= 1
	else:
		prob = _kSmallestCDF(cur_return, num_returns_gauss, num_no_jumps)

		if prob <= prob_cut_off:
			jumps[return_idx] = True
			num_no_jumps -= 1
		else:
			num_returns_gauss += 1

	return jumps, num_returns_gauss, num_no_jumps


def get_kSmallestCDF_single_side_inc(last_return: float, k_gauss: int, num_no_jumps: int, prob_cut_off: float):
	prob = _kSmallestCDF(last_return, k_gauss, num_no_jumps)
	is_jump = False

	if prob <= prob_cut_off:
		is_jump = True
		num_no_jumps -= 1
	else:
		k_gauss += 1

	return is_jump, k_gauss, num_no_jumps


def _kSmallestCDF(x, k, n):
	y = sp.betainc(np.double(k), n - np.double(k) + 1., stats.norm.cdf(x, 0., 1.))
	return y


# def FindTestJumps_original(results_df: pd.DataFrame, split_idx: int, halflife: int, halflife_multiplier: int, debug: bool = False):
# 	spread = results_df['spread_val_returns'].to_numpy()
#
# 	# if debug:
# 	# 	slice = spread[split_idx:]
# 	# 	vols, jumps, returns_adjusted = detect_jumps(slice, halflife, prob_cut_off=0.05, max_iter=100)
# 	# 	return jumps
#
# 	bandwidth = halflife * halflife_multiplier
# 	# start_idx = 0
# 	start_idx = max(split_idx - halflife * halflife_multiplier, 0)
# 	# slice = spread[start_idx:split_idx]
#
# 	# vols, jumps, ret_sorted, num_gauss_up_returns, num_gauss_down_returns, num_no_jumps = detect_jumps(slice, bandwidth, prob_cut_off=0.05, max_iter=100)
#
# 	#
# 	# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
# 	# X = np.array(range(len(slice)))
# 	# axes.plot(X, slice)
# 	# axes.set_title('Original cold start')
# 	# # axes.set_ylim([-100, 100])
# 	# axes.scatter(X[jumps], slice[jumps], color='red', marker='*')
# 	# fig.show()
#
# 	test_jumps = []
# 	k = split_idx + 1
# 	while k <= len(spread):
# 		slice = spread[start_idx:k]
# 		# vols, jumps, returns_adjusted = detect_jump_inc(slice, jumps, vols, bandwidth, prob_cut_off=0.05, max_iter=100)
# 		vols, jumps, ret_sorted, num_gauss_up_returns, num_gauss_down_returns, num_no_jumps, up_jumps, down_jumps = detect_jumps(slice,
# 																																 bandwidth,
# 																																 prob_cut_off=0.05,
# 																																 max_iter=100)
# 		test_jumps.append(jumps[-1])
# 		k += 1
#
# 	# jumps_concat = np.concatenate([jumps[:len(jumps) - len(test_jumps)], test_jumps])
# 	# slice = spread[start_idx:]
# 	# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
# 	# X = np.array(range(len(slice)))
# 	# axes.plot(X, slice)
# 	# axes.set_title(f'Original incremental. sum_jumps={sum(jumps)} sum_test_jumps={sum(test_jumps)} sum_jumps_concat={sum(jumps_concat)}')
# 	# # axes.set_ylim([-100, 100])
# 	# axes.scatter(X[jumps_concat], slice[jumps_concat], color='red', marker='*')
# 	# fig.show()
#
# 	return test_jumps
def find_jumps_incremental(returns: np.ndarray, split_idx: int, bandwidth: int):
	slice = returns[:split_idx]

	vols, jumps, ret_sorted, up_jumps, down_jumps = detect_jumps(slice, bandwidth, 0.01, max_iter=100)

	# train_jumps = jumps
	# test_jumps = []
	k = split_idx + 1
	tree_ret_sorted = SortedList(ret_sorted)
	prev_vol = vols[-1]

	while k <= len(returns):
		slice = returns[:k]
		last_vol, jumps, tree_ret_sorted, up_jumps, down_jumps = detect_jump_inc_alternative(slice,
																							 jumps,
																							 prev_vol,
																							 tree_ret_sorted,
																							 bandwidth,
																							 up_jumps,
																							 down_jumps,
																							 0.01)

		prev_vol = last_vol
		# test_jumps.append(jumps[-1])
		k += 1

	return jumps

def days_window_to_periods(data: pd.DataFrame, window_days: int):
	delta = timedelta(days=window_days)
	end_date = data.index[0] + delta

	df_slice = data[data.index <= end_date]
	window_periods = len(df_slice)

	return window_periods


def DetectResampledJumps(col_series: pd.Series, column_name: str, resample_frequency: str, volatility_window_days: int, train_days: int):
	column = col_series.resample(resample_frequency, label='right', closed='right').last()

	col_minvalue = column.min()

	# This is done to avoid crossing zero and as a result possible very high return values when actual values are close to 0
	if col_minvalue < 0:
		column = column + abs(column.min()) + 1

	returns = column.pct_change().fillna(0)
	returns_ar = returns.to_numpy()

	train_split_idx = days_window_to_periods(column, train_days)
	volatility_window = days_window_to_periods(column, volatility_window_days)

	jumps = find_jumps_incremental(returns_ar, train_split_idx, volatility_window)
	signed_jumps = jumps * np.sign(returns_ar)

	column_prefix = f'{column_name}_{resample_frequency}_{volatility_window_days}'
	jumps_df = pd.DataFrame(jumps.astype(int), columns=[f'{column_prefix}_jumps'], index=column.index)
	jumps_df[f'{column_prefix}_signed_jumps'] = signed_jumps

	return jumps_df


def FillJumpsDecreasing(series: np.ndarray):
	non_null_idx = np.flatnonzero(~np.isnan(series))

	for i in range(len(non_null_idx) - 1):
		start_i = non_null_idx[i]
		end_i = non_null_idx[i + 1]
		v0 = series[start_i]

		gap_len = end_i - start_i - 1
		if gap_len <= 0: continue
		if v0 == 1:
			# create a ramp from 1 down to 0 over gap_len+1 steps, take the first gap_len points
			ramp = np.linspace(1, 0, gap_len + 2)[1:-1]
			series[start_i + 1: end_i] = ramp
		else:
			series[start_i + 1: end_i] = 0
	return series


def MergeJumps(feats_df: pd.DataFrame, jumps_df: pd.DataFrame):
	merged = pd.merge_asof(feats_df, jumps_df, left_index=True, right_index=True, direction='backward')

	# Step 2: compute, for each A-index, which B-index it actually matched
	jumps_idx = jumps_df.index
	# for each timestamp in merged.index, find position in B.index
	#  >0 means there is a match; position-1 is the matched B index
	positions = np.searchsorted(jumps_idx, merged.index, side='right') - 1
	# build a Series of matched times (NaT where pos < 0)
	matched_mask = np.where(positions >= 0, jumps_idx[positions], pd.NaT)
	matched_times = pd.Series(matched_mask, index=merged.index)

	# Step 3: flag only the *first* row of each matched-time block
	is_new_block = matched_times.ne(matched_times.shift()) & matched_times.notna()

	jumps_columns = list(jumps_df.columns)
	# Step 4: keep b_val only on those first rows, else NaN
	merged[jumps_columns] = merged[jumps_columns].where(is_new_block)

	return merged
