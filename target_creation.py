import logging

import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import argrelmax, argrelmin

from utils.data_structures import SignalTypes


def set_extrem_shifted_indices(result_length: int, indices_loc: set, numNeighbours: int) -> set[int]:
	indices = set()

	for i in range(1, numNeighbours + 1):
		tmp_indices_down = set([idx - i for idx in indices_loc if idx - i > 0])
		tmp_indices_up = set([idx + i for idx in indices_loc if idx + i < result_length])
		indices.update(tmp_indices_down)
		indices.update(tmp_indices_up)

	return indices


def AddPeakNeighboursSingleColumn(feats_df: pd.DataFrame,
								  combination: tuple[str, str],
								  target_col: str, period: int,
								  resulting_target_column: str,
								  numNeighbours: int) -> pd.DataFrame:
	# logging.info(f'Start target creation for {combination}')

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
