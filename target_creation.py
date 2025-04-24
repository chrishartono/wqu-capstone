import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import argrelmax, argrelmin


def set_extrem_shifted_indices(result_length: int, indices_loc: set, numNeighbours: int) -> set[int]:
	indices = set()

	for i in range(1, numNeighbours + 1):
		tmp_indices_down = set([idx - i for idx in indices_loc if idx - i > 0])
		tmp_indices_up = set([idx + i for idx in indices_loc if idx + i < result_length])
		indices.update(tmp_indices_down)
		indices.update(tmp_indices_up)

	return indices


def AddPeakNeighboursSingleColumn(feats_df: pd.DataFrame, target_col: str, period: int, resulting_target_column: str, numNeighbours: int) -> pd.DataFrame:

	results = feats_df.reset_index(drop=True)

	results[resulting_target_column] = 0
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
	results.loc[list(max_all_indices_set), resulting_target_column] = 2
	results.loc[list(min_all_indices_set), resulting_target_column] = 1

	results.index = feats_df.index

	results[resulting_target_column].fillna(0, inplace=True)

	fig, axes = plt.subplots(nrows=2, ncols=1)
	results[target_col].plot(ax=axes[0])
	axes[0].scatter(results.index[results[resulting_target_column] != 0], results.loc[results[resulting_target_column] != 0][target_col], color='red', marker='*')
	plt.show()

	return results
