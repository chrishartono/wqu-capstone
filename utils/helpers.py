from datetime import timedelta
import pandas as pd


def DaysWindowToPeriods(data: pd.DataFrame, window_days: int):
	rolling_delta = timedelta(days=window_days)
	rolling_window_end_date = data.index[0] + rolling_delta
	df_slice = data[data.index <= rolling_window_end_date]

	window_periods = len(df_slice)
	return window_periods