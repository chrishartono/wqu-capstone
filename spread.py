import pandas as pd
import statsmodels.api as sm

def BuildSpread(prices_df: pd.DataFrame, combination: tuple[str, str]) -> tuple[pd.DataFrame, float]:
	"""
	This function creates spread dataframe with OLS for train set.

	:param prices_df: Pandas DataFrame with 2 columns (one for each time series).
	:param combination: Tuple with name of each crypto pair. Should match prices_df column names.
	:return: Tuple of spread dataframe and combination tuple (for using with parallelization).
	"""

	Y = prices_df[combination[0]]
	X = prices_df[combination[1]]
	X = sm.add_constant(X)
	model = sm.OLS(Y, X)
	results = model.fit_regularized()
	coef = results.params[1]

	spread = pd.DataFrame(Y - X * coef, columns=['spread'], index=prices_df.index)

	return spread, coef

def UpdateSpread(prices_df: pd.DataFrame, combination: tuple[str, str], coef: float) -> tuple[pd.DataFrame]:
	"""
	This function creates spread dataframe with coefficient given as parameter.

	:param prices_df: Pandas DataFrame with 2 columns (one for each time series).
	:param combination: Tuple with name of each crypto pair. Should match prices_df column names.
	:param coef: spread coefficient.
	:return: Tuple of spread dataframe and combination tuple (for using with parallelization).
	"""

	Y = prices_df[combination[0]]
	X = prices_df[combination[1]]
	spread = pd.DataFrame(Y - X * coef, columns=['spread'], index=prices_df.index)

	return spread