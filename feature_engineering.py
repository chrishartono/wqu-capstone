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

def add_zscores(train: pd.DataFrame, test: pd.DataFrame):
	for col in train.columns:
		mean = train[col].mean()
		std = train[col].std()
		train[f'{col}_zscore'] = (train[col] - mean) / std
		test[f'{col}_zscore'] = (test[col] - mean) / std

	return train, test

def add_rolling_hurst(train: pd.DataFrame, test: pd.DataFrame, rolling_window_days: int):
	hurst_columns = ['spread']

	df = pd.concat([train, test], axis=0)
	rolling_delta = timedelta(days=rolling_window_days)
	rolling_window_end_date = df.index[0] + rolling_delta
	df_slice = df[df.index <= rolling_window_end_date]

	rolling_window_periods = len(df_slice)
	for col in hurst_columns:
		df[f'{col}_hurst'] = df[col].rolling(window=rolling_window_periods).apply(lambda x: compute_Hc(x, kind='price', simplified=True)[0], raw=True)

	train_len = len(train)
	train = df.iloc[:train_len]
	test = df.iloc[train_len:]

	return train, test

def clean(df: pd.DataFrame):
	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df.ffill(inplace=True)
	df.fillna(0)

	return df

def AddFeatures(train: pd.DataFrame, test: pd.DataFrame, combination: tuple[str, str], rolling_window_days: int):
	train = add_basic_features(train, combination)
	test = add_basic_features(test, combination)

	train, test = add_zscores(train, test)
	# train, test = add_rolling_hurst(train, test, rolling_window_days)

	train = clean(train)
	test = clean(test)

	return train, test
