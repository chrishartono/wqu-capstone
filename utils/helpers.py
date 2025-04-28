import logging
from datetime import timedelta
import pandas as pd


def DaysWindowToPeriods(data: pd.DataFrame, window_days: int):
	rolling_delta = timedelta(days=window_days)
	rolling_window_end_date = data.index[0] + rolling_delta
	df_slice = data[data.index <= rolling_window_end_date]

	window_periods = len(df_slice)
	return window_periods

def LogValueCounts(values, counts, data_type, data_length):
	countByValue = {values[i]: counts[i] for i in range(len(counts))}

	msg = f'{data_type} '
	for value in sorted(countByValue.keys()):
		msg += f'{int(value)}={countByValue[value] / data_length * 100:0.1f}%, '

	logging.info(msg.rstrip(', '))