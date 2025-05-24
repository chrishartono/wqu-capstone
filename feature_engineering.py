import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from hurst import compute_Hc
from pmdarima.arima import auto_arima, ADFTest

from utils.helpers import DaysWindowToPeriods


def add_basic_features(feats_df: pd.DataFrame, combination: tuple[str, str]):
	columns_for_returns = ['close', 'volume', 'close-open', 'high-low']

	df = feats_df.copy()
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

def add_zscores(feats_df: pd.DataFrame, window_period: int):
	data = feats_df.copy()

	for col in data.columns:
		zscore_colname = f'{col}_zscore_{window_period}'
		zscore_extrem_colname = f'{col}_zscore_extrem_{window_period}'

		mean = data[col].rolling(window=window_period).mean()
		std = data[col].rolling(window=window_period).std()
		data[zscore_colname] = (data[col] - mean) / std

		data[f'{col}_10q'] = data[zscore_colname].rolling(window=window_period).quantile(0.1)
		data[f'{col}_90q'] = data[zscore_colname].rolling(window=window_period).quantile(0.9)
		data[zscore_extrem_colname] = 0
		data.loc[data[zscore_colname] < data[f'{col}_10q'], zscore_extrem_colname] = -1
		data.loc[data[zscore_colname] > data[f'{col}_90q'], zscore_extrem_colname] = 1

		data.drop([f'{col}_10q', f'{col}_90q'], axis=1, inplace=True)
	return data

def add_rolling_hurst(feats_df: pd.DataFrame, window_period: int):
	hurst_columns = ['spread']

	data = feats_df.copy()

	for col in hurst_columns:
		hurst_colname = f'{col}_hurst_{window_period}'
		data[f'{col}_shifted'] = data[col] + abs(data[col].min()) + 1
		try:
			data[hurst_colname] = data[f'{col}_shifted'].rolling(window=window_period).apply(lambda x: compute_Hc(x, kind='price', simplified=True)[0], raw=True)
		except:
			data[hurst_colname] = 0.5

		data.drop([f'{col}_shifted'], axis=1, inplace=True)

	return data

def add_arima(feats_df: pd.DataFrame, end_train_date: datetime):
	train = feats_df.loc[feats_df.index <= end_train_date, 'spread']
	test = feats_df[feats_df.index > end_train_date, 'spread']

	adf_test = ADFTest(alpha=0.05)
	should_diff = adf_test.should_diff(train)

	arima_model =	auto_arima(train,
								start_p=0,
								d=1,
								start_q=1,
								max_p=5,
								max_d=5,
								start_P=0,
								D=1,
								start_Q=0,
								max_P=5,
								max_D=5,
								max_Q=5,
								m = 12,
								seasonal = True,
								error_action='warn',
								trace = True,
								supress_warnings = True,
								stepwise = True,
								random_state=42,
								n_fits=50)

def add_catboost_spread_prediction(feats_df: pd.DataFrame, end_train_date: datetime):
	train = feats_df.loc[feats_df.index <= end_train_date]
	test = feats_df[feats_df.index > end_train_date]

	X_train = train.drop(columns=['spread'], axis=1)
	X_test = test.drop(columns=['spread'], axis=1)
	y_train = train['spread']

	catboost_hyperparameters = {'depth': 3, 'iterations': 100, 'learning_rate': 0.1, 'thread_count':1}
	clf = CatBoostRegressor(verbose=0, **catboost_hyperparameters)
	clf.fit(X=X_train, y=y_train)

	train['spread_prediction'] = clf.predict(X_train)
	test['spread_prediction'] = clf.predict(X_test)

	modified_feats_df = pd.concat([train, test], axis=0)

	return modified_feats_df

def add_spread_above_pred(feats_df: pd.DataFrame):
	local_feats_df = feats_df.copy()
	local_feats_df['spread_above_prediction'] = 0
	local_feats_df.loc[local_feats_df['spread'] > local_feats_df['spread_prediction'], 'spread_above_prediction'] = 1

	return local_feats_df

def clean(feats_df: pd.DataFrame):
	df = feats_df.copy()

	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df.ffill(inplace=True)
	df.fillna(0, inplace=True)

	df.dropna(inplace=True)

	return df

def AddFeatures(feats_df: pd.DataFrame, combination: tuple[str, str], rolling_windows_days_list: list[int], end_train_date: datetime) -> pd.DataFrame:
	# logging.info(f'Start adding features for {combination}')

	data = feats_df.copy()

	data = add_basic_features(data, combination)
	data = add_catboost_spread_prediction(data, end_train_date)

	for rolling_window_days in rolling_windows_days_list:
		window_periods = DaysWindowToPeriods(data, rolling_window_days)
		data = add_zscores(data, window_periods)
		data = add_rolling_hurst(data, window_periods)

	data = add_spread_above_pred(data)
	data = clean(data)

	return data
