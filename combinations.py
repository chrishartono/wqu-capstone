import logging
from itertools import combinations

import pandas as pd
from arch.unitroot._phillips_ouliaris import PhillipsOuliarisTestResults
from joblib import Parallel, delayed
from tqdm import tqdm

from comovement import ComovementType, TestCombinationComovement


def CreateAllPossibleCombinations(prices_df: pd.DataFrame):
	logging.info(f'Start creating combinations for {len(prices_df.columns)} pairs: {list(prices_df.columns)}')

	pair_combinations = []
	combs_set = set()
	close_columns = [col for col in prices_df.columns if 'close' in col]
	for c0, c1 in combinations(close_columns, r=2):
		if (c0, c1) in combs_set: continue

		combs_set.add((c0, c1))
		combs_set.add((c1, c0))
		pair_combinations.append((c0, c1))
		pair_combinations.append((c1, c0))

	logging.info(f'Created total number of combinations: {len(pair_combinations)}')
	return pair_combinations


def SearchForGoodCombinations(prices_df: pd.DataFrame, all_possible_combinations: list, comovement_type: ComovementType, n_jobs: int) \
		-> list[tuple[tuple[str, str], PhillipsOuliarisTestResults]]:
	logging.info(f'Start searching for good combinations from total of {len(all_possible_combinations)} possible combinations '
				 f'from {prices_df.index[0]} to {prices_df.index[-1]} and for pairs: {list(prices_df.columns)}')

	params = [(prices_df[list(combination)], combination, comovement_type) for combination in all_possible_combinations]
	results = (Parallel(n_jobs=n_jobs, prefer="processes")(delayed(TestCombinationComovement)(*p)
														   for p in tqdm(params, total=len(params), desc="Combinations search:")))
	# results = (Parallel(n_jobs=n_jobs, prefer="processes")(delayed(TestCombinationComovement)(*p) for p in params))

	all_good_combinations = [(comb, coint_vector) for isGood, comb, coint_vector in results if isGood and coint_vector is not None]
	good_combinations = []
	used_pairs = set()
	for comb, coint_vector in all_good_combinations:
		if comb[0] not in used_pairs and comb[1] not in used_pairs:
			good_combinations.append((comb, coint_vector))

		used_pairs.add(comb[0])
		used_pairs.add(comb[1])

	logging.info(f'Finally got {len(good_combinations)} good combinations for prices slice from {prices_df.index[0]} to {prices_df.index[-1]}')

	return good_combinations
