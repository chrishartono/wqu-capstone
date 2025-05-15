import logging
from enum import IntEnum
from unittest import result

import numpy as np
import pandas as pd
from arch.unitroot._phillips_ouliaris import PhillipsOuliarisTestResults
from arch.unitroot.cointegration import phillips_ouliaris
from statsmodels.tsa.stattools import grangercausalitytests

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ComovementType(IntEnum):
	COINTEGRATION = 0
	GRANGER_CAUSALITY = 1

def test_cointegration(prices_df: pd.DataFrame, combination: tuple[str, str], significance_level=0.05) -> \
		tuple[bool, tuple[str, str], PhillipsOuliarisTestResults]:
	"""
	This function checks for cointegration between 2 time series with Phillips-Ouliaris Zt and Za tests

	:param prices_df: Pandas DataFrame with 2 columns (one for each time series).
	:param combination: Tuple with name of each crypto pair. Should match prices_df column names.
	:param significance_level: Significance level against which we check our p_value.
	:return: Tuple of Boolean flag indicating cointegration and combination tuple (for using with parallelization).
	"""
	try:
		result = phillips_ouliaris(prices_df[combination[0]], prices_df[combination[1]], trend="ct", test_type="Zt")
	except:
		return False, combination, None

	coint_vector = result.cointegrating_vector

	# po_pvalues = [phillips_ouliaris(prices_df[combination[0]], prices_df[combination[1]], trend="ct", test_type="Zt").pvalue,
	# 			  phillips_ouliaris(prices_df[combination[0]], prices_df[combination[1]], trend="ct", test_type="Za").pvalue,
	# 			  phillips_ouliaris(prices_df[combination[0]], prices_df[combination[1]], trend="ct", test_type="Pu").pvalue,
	# 			  phillips_ouliaris(prices_df[combination[0]], prices_df[combination[1]], trend="ct", test_type="Pz").pvalue]
	# cointegrated = np.all([pvalue < significance_level for pvalue in po_pvalues])
	cointegrated = result.pvalue < significance_level
	if not cointegrated:
		logging.info(f'{combination} is not cointegrated. pvalue: {result.pvalue}')
		return False, combination, coint_vector

	return True, combination, coint_vector

def test_granger_causality(prices_df: pd.DataFrame, combination: tuple[str, str], significance_level=0.05) -> \
		tuple[bool, tuple[str, str], PhillipsOuliarisTestResults]:
	"""
	This function checks if time series 1 first differences are Granger Caused by time series 2 first differences.

	:param prices_df: Pandas DataFrame with 2 columns (one for each time series).
	:param combination: Tuple with name of each crypto pair. Should match prices_df column names.
	:param significance_level: Significance level against which we check our p_value.
	:return: Tuple of Boolean flag indicating Granger Causality and combination tuple (for using with parallelization).
	"""

	cointegrated, _, coint_vector = test_cointegration(prices_df, combination, significance_level)
	if not cointegrated:
		logging.info(f'Will not test Granger causality further for {combination} because not cointegrated.')
		return False, combination, coint_vector

	diffs = prices_df.diff(1).dropna()
	try:
		gc_res = grangercausalitytests(diffs, maxlag=10, verbose=False)
	except:
		return False, combination, coint_vector

	suitable_lags = []
	for lag in gc_res.values():
		tests = lag[0]
		pvalues = [tup[1] for tup in tests.values()]
		suitable_lags.append(np.any([pvalue  < significance_level for pvalue in pvalues]))

	granger_caused = np.any(suitable_lags)

	if not granger_caused:
		logging.info(f'{combination[1]} does NOT cause {combination[0]}')
		return False, combination, coint_vector

	return True, combination, coint_vector

def TestCombinationComovement(prices_df: pd.DataFrame, combination: tuple[str, str], comovement_type: ComovementType, significance_level=0.05) -> \
		tuple[bool, tuple[str, str], PhillipsOuliarisTestResults]:
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