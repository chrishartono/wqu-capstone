import numpy as np
import pandas as pd
from enum import IntEnum
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

class TopModelType(IntEnum):
	HMM = 0
	ARIMA = 1

class TopModelArima:
	def __init__(self, pre_data: pd.DataFrame, 
			  window: int = 24, reference_column: str = 'spread'):

		self.__window = window
		self.__pre_data = pre_data
		self.__reference_column = reference_column

		self.__std_history = self.backfill_std_history(pre_data, window)

	def backfill_std_history(self, pre_data: pd.DataFrame, window: int):
		history = []

		for i in tqdm(range(window, len(pre_data)), desc='backfilling ARIMA std history'):
			model_ = ARIMA(pre_data.iloc[i - window:i][self.__reference_column], order=(1, 0, 1))
			res = model_.fit()
			std = np.std(res.resid)
			history.append(std)

		return history

	def predict(self, data: pd.DataFrame):
		pred = []

		for i in tqdm(range(len(data)), desc='generating ARIMA residual analysis'):
			if i < self.__window:
				x = pd.concat([
						self.__pre_data.iloc[-self.__window + i:],
						data.iloc[:i]
					], axis=0)[self.__reference_column]
			else:
				x = data.iloc[i - self.__window:i][self.__reference_column]

			model_ = ARIMA(x, order=(1, 0, 1))
			res = model_.fit()
			std = np.std(res.resid)

			# pause the trade if the spread crosses the upper threshold
			threshold = np.quantile(self.__std_history, q=0.75)
			pred.append(1 if std < threshold else 0)

			self.__std_history.append(std)

		return pred
	
class TopModelHMM:
	pass