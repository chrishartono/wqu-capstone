import logging
from enum import IntEnum

import numpy as np
import pandas as pd
from arch.unitroot.cointegration import phillips_ouliaris
from statsmodels.tsa.stattools import grangercausalitytests


class ComovementType(IntEnum):
	COINTEGRATION = 0
	GRANGER_CAUSALITY = 1

def test_cointegration(prices_df: pd.DataFrame, combination: tuple[str, str], significance_level=0.05) -> tuple[bool, tuple[str, str]]:
	"""
	This function checks for cointegration between 2 time series with Phillips-Ouliaris Zt and Za tests

	:param prices_df: Pandas DataFrame with 2 columns (one for each time series).
	:param combination: Tuple with name of each crypto pair. Should match prices_df column names.
	:param significance_level: Significance level against which we check our p_value.
	:return: Tuple of Boolean flag indicating cointegration and combination tuple (for using with parallelization).
	"""
	result = phillips_ouliaris(prices_df[combination[0]], prices_df[combination[1]], trend="ct", test_type="Zt")
	v = result.cointegrating_vector
	spread = prices_df[combination[0]]*v[combination[0]] + prices_df[combination[1]]*v[combination[1]]

	po_pvalues = [phillips_ouliaris(prices_df[combination[0]], prices_df[combination[1]], trend="ct", test_type="Zt").pvalue,
				  phillips_ouliaris(prices_df[combination[0]], prices_df[combination[1]], trend="ct", test_type="Za").pvalue,
				  phillips_ouliaris(prices_df[combination[0]], prices_df[combination[1]], trend="ct", test_type="Pu").pvalue,
				  phillips_ouliaris(prices_df[combination[0]], prices_df[combination[1]], trend="ct", test_type="Pz").pvalue]
	cointegrated = np.all([pvalue < significance_level for pvalue in po_pvalues])
	if not cointegrated:
		logging.info(f'{combination} is not cointegrated. po_pvalues: {po_pvalues}')
		return False, combination

	return True, combination

def test_granger_causality(prices_df: pd.DataFrame, combination: tuple[str, str], significance_level=0.05) -> tuple[bool, tuple[str, str]]:
	"""
	This function checks if time series 1 first differences are Granger Caused by time series 2 first differences.

	:param prices_df: Pandas DataFrame with 2 columns (one for each time series).
	:param combination: Tuple with name of each crypto pair. Should match prices_df column names.
	:param significance_level: Significance level against which we check our p_value.
	:return: Tuple of Boolean flag indicating Granger Causality and combination tuple (for using with parallelization).
	"""

	diffs = prices_df.diff(1).dropna()
	gc_res = grangercausalitytests(diffs, maxlag=10)

	gc_pvalues = [v[1] for v in gc_res.values()]
	granger_caused = np.any([pvalue < significance_level for pvalue in gc_pvalues])

	if not granger_caused:
		logging.info(f'{combination[1]} does NOT cause {combination[0]}. gc_pvalues for lags: {gc_pvalues}')
		return False, combination

	return True, combination

def TestCombinationComovement(prices_df: pd.DataFrame, combination: tuple[str, str], comovement_type: ComovementType, significance_level=0.05) \
		-> tuple[bool, tuple[str, str]]:
	"""
	This function checks if time series 1 first differences are Granger Caused by time series 2 first differences.

	:param prices_df: Pandas DataFrame with 2 columns (one for each time series).
	:param combination: Tuple with name of each crypto pair. Should match prices_df column names.
	:param comovement_type: Type of method to choose for testing for comevement.
	:param significance_level: Significance level against which we check our p_value.
	:return: Tuple of Boolean flag indicating Granger Causality and combination tuple (for using with parallelization).
	"""

	if comovement_type == ComovementType.COINTEGRATION:
		return test_cointegration(prices_df, combination)
	elif comovement_type == ComovementType.GRANGER_CAUSALITY:
		return test_granger_causality(prices_df, combination)
	else:
		raise(f'Unknown comovement type: {comovement_type}')