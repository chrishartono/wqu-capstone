import numpy as np
import pandas as pd
from statsmodels.regression.rolling import RollingOLS

import warnings

from utils.helpers import DaysWindowToPeriods

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	import pandas as pd
pd.options.mode.chained_assignment = None

from arch.unitroot._phillips_ouliaris import PhillipsOuliarisTestResults

def AddPriceChangeSpread(prices_df: pd.DataFrame, combination: tuple[str, str]):
	c0 = combination[0]
	c1 = combination[1]

	data = prices_df.copy()
	data[f'{c0}_returns'] = data[c0].pct_change()
	data[f'{c1}_returns'] = data[c1].pct_change()
	data['spread'] = data[f'{c0}_returns'] / data[f'{c1}_returns']

	data.drop([f'{c0}_returns', f'{c1}_returns'], inplace=True, axis=1)

	return data

def AddRollingOLSSpread(prices_df: pd.DataFrame, combination: tuple[str, str], window_days: int):
	c0 = combination[0]
	c1 = combination[1]

	data = prices_df.copy()
	window_periods = DaysWindowToPeriods(data, window_days)

	model_rols = RollingOLS(endog=data[c0], exog=data[c1], window=window_periods, min_nobs=window_periods)
	results_rols = model_rols.fit()

	hedge_ratio = results_rols.params[c1].to_numpy()
	data['spread'] = (data[c0] - data[c1] * hedge_ratio)

	data.dropna(inplace=True, axis=0, subset='spread')
	hedge_ratio = hedge_ratio[~np.isnan(hedge_ratio)]

	c0_coef = np.full(len(data), fill_value=1)
	coefs_vector = {c0: c0_coef, c1: -hedge_ratio}

	return data, coefs_vector

def AddCointCoefSpread(prices_df: pd.DataFrame, combination: tuple[str, str], coint_vector: PhillipsOuliarisTestResults):
	c0 = combination[0]
	c1 = combination[1]

	data = prices_df.copy()
	data['spread'] = prices_df[c0]*coint_vector[c0] + prices_df[c1]*coint_vector[c1]

	return data

def AddPolyfitSpread(prices_df: pd.DataFrame, combination: tuple[str, str], coefs: np.ndarray = None) -> tuple[pd.DataFrame, np.ndarray]:
	"""
	This function creates spread dataframe with OLS for train set.

	:param prices_df: Pandas DataFrame with 2 columns (one for each time series).
	:param combination: Tuple with name of each crypto pair. Should match prices_df column names.
	:return: Tuple of spread dataframe and coefficients.
	"""

	Y = prices_df[combination[0]]
	X = prices_df[combination[1]]

	if coefs is None: coefs = np.polyfit(X, Y, 3)
	Y_fit = np.polyval(coefs, X)

	prices_df['spread'] = Y - Y_fit

	return prices_df, coefs