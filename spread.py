import numpy as np
import pandas as pd

import warnings

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	import pandas as pd
pd.options.mode.chained_assignment = None

from arch.unitroot._phillips_ouliaris import PhillipsOuliarisTestResults


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