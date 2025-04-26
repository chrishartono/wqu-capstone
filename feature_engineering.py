from datetime import timedelta

import numpy as np
import pandas as pd
from hurst import compute_Hc

def add_basic_features(df: pd.DataFrame, combination: tuple[str, str]):
	columns_for_returns = ['close', 'volume', 'close-open', 'high-low']

	pairs = set([c.split('_')[1] for c in combination])

	for pair in pairs:
		df[f'close-open_{pair}'] = (df[f'close_{pair}'] - df[f'open_{pair}']) / df[f'open_{pair}']
		df[f'high-low_{pair}'] = (df[f'high_{pair}'] - df[f'low_{pair}']) / df[f'low_{pair}']

		for col_ret in columns_for_returns:
			df[f'{col_ret}_{pair}_returns'] = df[f'{col_ret}_{pair}'].pct_change().fillna(0)

	# Can't use pct_change here because spread may have negative values
	df['spread_returns'] = (df['spread'] - df['spread'].shift(1)) / abs(df['spread'].shift(1))
	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df.fillna(0, inplace=True)

	return df

def add_zscores(data: pd.DataFrame, window_periods: int):
	for col in data.columns:
		mean = data[col].rolling(window=window_periods).mean()
		std = data[col].rolling(window=window_periods).std()
		data[f'{col}_zscore'] = (data[col] - mean) / std

	return data

def add_rolling_hurst(data: pd.DataFrame, window_periods: int):
	hurst_columns = ['spread']

	for col in hurst_columns:
		data[f'{col}_shifted'] = data[col] + abs(data[col].min()) + 1
		data[f'{col}_hurst'] = data[f'{col}_shifted'].rolling(window=window_periods).apply(lambda x: compute_Hc(x, kind='price', simplified=True)[0], raw=True)

	return data

def clean(df: pd.DataFrame):
	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df.ffill(inplace=True)
	df.fillna(0)

	return df

def AddFeatures(train: pd.DataFrame, test: pd.DataFrame, combination: tuple[str, str], rolling_window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
	data = pd.concat([train, test], axis=0)

	rolling_delta = timedelta(days=rolling_window_days)
	rolling_window_end_date = data.index[0] + rolling_delta
	df_slice = data[data.index <= rolling_window_end_date]

	window_periods = len(df_slice)

	data = add_basic_features(data, combination)
	data = add_zscores(data, window_periods)
	data = add_rolling_hurst(data, window_periods)

	data = clean(data)
	train = data.iloc[:len(train)]
	test = data.iloc[len(train):]

	return train, test
