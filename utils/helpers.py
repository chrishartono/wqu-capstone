import logging
from datetime import timedelta

import numpy as np
import pandas as pd


def DaysWindowToPeriods(data: pd.DataFrame, window_days: int):
	rolling_delta = timedelta(days=window_days)
	rolling_window_end_date = data.index[0] + rolling_delta

	window_periods = len(data[data.index <= rolling_window_end_date])

	return window_periods


def LogValueCounts(values, counts, data_type, data_length):
	countByValue = {values[i]: counts[i] for i in range(len(counts))}

	msg = f'{data_type} '
	for value in sorted(countByValue.keys()):
		msg += f'{int(value)}={countByValue[value] / data_length * 100:0.1f}%, '

	logging.info(msg.rstrip(', '))


def SemiStd(series):
	average = np.nanmean(series)
	r_below = series[series < average]
	if (len(r_below) == 0): return 0.001
	return np.sqrt(1 / len(r_below) * np.sum((average - r_below) ** 2))
