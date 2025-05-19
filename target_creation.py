import logging
from enum import IntEnum

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import argrelmax, argrelmin
import statsmodels.api as sm

from utils.data_structures import SignalTypes
from utils.helpers import DaysWindowToPeriods


class TargetType(IntEnum):
	PEAK_NEIGHBOURS_CLF = 0
	OLS_CLF = 1


def set_extrem_shifted_indices(result_length: int, indices_loc: set, numNeighbours: int) -> set[int]:
	indices = set()

	for i in range(1, numNeighbours + 1):
		tmp_indices_down = set([idx - i for idx in indices_loc if idx - i > 0])
		tmp_indices_up = set([idx + i for idx in indices_loc if idx + i < result_length])
		indices.update(tmp_indices_down)
		indices.update(tmp_indices_up)

	return indices


def AddPeakNeighboursTarget(feats_df: pd.DataFrame,
							combination: tuple[str, str],
							target_col: str,
							resulting_target_column: str,
							target_params: dict) -> pd.DataFrame:
	# logging.info(f'Start peak neighbours target creation for {combination}')

	numNeighbours = target_params['numNeighbours']
	period = DaysWindowToPeriods(feats_df, target_params['rolling_window_days'])

	results = feats_df.reset_index(drop=True)

	results[resulting_target_column] = SignalTypes.NONE
	result_length = len(results)

	max_indices = set(argrelmax(feats_df[target_col].values, order=period)[0])
	max_surrounding_indices_set = set_extrem_shifted_indices(result_length, max_indices, numNeighbours)
	max_all_indices_set = set(max_indices) | max_surrounding_indices_set

	min_indices = set(argrelmin(feats_df[target_col].values, order=period)[0])
	min_surrounding_indices_set = set_extrem_shifted_indices(result_length, min_indices, numNeighbours)
	min_all_indices_set = set(min_indices) | min_surrounding_indices_set

	intersection = min_all_indices_set & max_all_indices_set
	max_all_indices_set -= intersection
	min_all_indices_set -= intersection
	results.loc[list(max_all_indices_set), resulting_target_column] = SignalTypes.SELL
	results.loc[list(min_all_indices_set), resulting_target_column] = SignalTypes.BUY

	results.index = feats_df.index

	results[resulting_target_column].fillna(0, inplace=True)

	# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
	# ax.plot(results.index, results[target_col], 'b-')
	# ax.scatter(results.index[results[resulting_target_column] == 1], results.loc[results[resulting_target_column] == 1][target_col], color='green',
	# marker='*', label='minima')
	# ax.scatter(results.index[results[resulting_target_column] == 2], results.loc[results[resulting_target_column] == 2][target_col], color='red',
	# marker='*', label='maxima')
	# ax.set_xlabel('Timestamp')
	# ax.set_ylabel('Returns')
	# ax.set_title(f'{combination} spread with points chosen as targets')
	# ax.legend()
	# plt.tight_layout()
	# fig.savefig(f'target.png', dpi=300)
	# plt.show()

	return results


def AddClassificationOLSTarget(feats_df: pd.DataFrame,
							   combination: tuple[str, str],
							   target_col: str,
							   resulting_target_column: str,
							   target_params: dict):
	# logging.info(f'Start classification OLS target creation for {combination}')

	look_ahead_days = target_params['look_ahead_days']
	reg_points_thresh_frac = target_params['reg_points_thresh_frac']

	window = look_ahead_days * 24
	values = feats_df[target_col].to_numpy()
	target = np.full(len(values), SignalTypes.NONE)
	x_values = np.arange(window)

	X = sm.add_constant(x_values)
	i = 0
	while i + window < len(values):
		y = values[i:i + window]
		result = sm.OLS(y, X).fit()
		intercept = result.params[0]
		slope = result.params[1]

		regression_line = x_values * slope + intercept
		current_value = values[i]
		regression_above_points = regression_line[regression_line > current_value]
		regression_below_points = regression_line[regression_line < current_value]

		above_points_frac = len(regression_above_points) / len(regression_line)
		below_points_frac = len(regression_below_points) / len(regression_line)

		if above_points_frac >= reg_points_thresh_frac:
			target[i] = SignalTypes.BUY
		elif below_points_frac >= reg_points_thresh_frac:
			target[i] = SignalTypes.SELL

		i += 1

	feats_target_df = feats_df.copy()
	feats_target_df[resulting_target_column] = target

	# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
	# ax.plot(feats_target_df.index, feats_target_df[target_col], 'b-')
	#
	# buys_mask = feats_target_df[resulting_target_column] == SignalTypes.BUY.value
	# sells_mask = feats_target_df[resulting_target_column] == SignalTypes.SELL.value
	#
	# ax.plot(feats_target_df.index[buys_mask], feats_target_df.loc[buys_mask, target_col], 'g^', markersize=4)
	# ax.plot(feats_target_df.index[sells_mask], feats_target_df.loc[sells_mask, target_col], 'rv', markersize=4)
	#
	# ax.set_xlabel('Timestamp')
	# ax.set_ylabel('Returns')
	# ax.set_title(f'{combination} spread with points chosen as targets')
	# ax.legend()
	# plt.tight_layout()
	# fig.savefig(f'{combination}_target.png', dpi=300)
	# plt.show()

	return feats_target_df
