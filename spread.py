import numpy as np
import pandas as pd
import statsmodels.api as sm

def AddSpread(prices_df: pd.DataFrame, combination: tuple[str, str], coefs: np.ndarray = None) -> tuple[pd.DataFrame, np.ndarray]:
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