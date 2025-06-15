#%%
import numpy as np
import pandas as pd
from enum import IntEnum
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
from hmmlearn import hmm
from spread import AddPolyfitSpread


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
			threshold = np.quantile(self.__std_history, q=0.9)
			pred.append(1 if std < threshold else 0)

			self.__std_history.append(std)

		return pred
	
#%%


class TopModelHMM:
    def __init__(self, pre_data: pd.DataFrame, 
                 window: int = 24, reference_column: str = 'spread', 
                 n_components: int = 2):
        self.__window = window
        self.__pre_data = pre_data
        self.__reference_column = reference_column
        self.__n_components = n_components

        # Fit initial HMM on pre_data
        self.model = self.fit_hmm(pre_data[reference_column].values.reshape(-1, 1))

    def fit_hmm(self, data):
        # Gaussian HMM, fit to the windowed spread TODO other tests
        model = hmm.GaussianHMM(n_components=self.__n_components, covariance_type="full", n_iter=100)
        model.fit(data)
        return model

    def predict(self, data: pd.DataFrame):
        pred = []
        all_data = pd.concat([self.__pre_data, data], axis=0)
        spread_series = all_data[self.__reference_column].values

        for i in tqdm(range(len(data)), desc='generating HMM regime analysis'):
            if i < self.__window:
                window_data = np.concatenate([
                    self.__pre_data[self.__reference_column].values[-self.__window + i:], 
                    data[self.__reference_column].values[:i]
                ]).reshape(-1, 1)
            else:
                window_data = data[self.__reference_column].values[i - self.__window:i].reshape(-1, 1)

            # Remove NaNs
            window_data = window_data[~np.isnan(window_data).flatten()].reshape(-1, 1)
            # Ensure enough data for HMM
            if len(window_data) < self.__n_components:
                pred.append(0)
                continue

            try:
                # Re-fit HMM
                hmm_model = self.fit_hmm(window_data)
                hidden_states = hmm_model.predict(window_data)
                means = hmm_model.means_.flatten()
                covars = hmm_model.covars_.flatten()
                low_var_state = np.argmin(covars)
                current_state = hidden_states[-1]
                pred.append(1 if current_state == low_var_state else 0)
            except Exception as e:
                # If fitting or prediction fails, skip this window
                pred.append(0)
                continue

        return pred

#test 

#%%
prices_df = pd.read_parquet('dataset/binance_1h_ohlcv_2021-2025.parquet')
prices_df = prices_df.set_index("date")

prices_df = prices_df[(prices_df.index >= '2022-01-01') & (prices_df.index <= '2024-09-01')]

#%%
from combinations import CreateAllPossibleCombinations
pairs = CreateAllPossibleCombinations(prices_df)




# %%
def AddPolyfitSpread(prices_df, combination, coefs=None, degree=1):
    df = prices_df.copy()
    Y = df[combination[0]].astype('float64')
    X = df[combination[1]].astype('float64')
    mask = np.isfinite(X) & np.isfinite(Y)
    X = X[mask]
    Y = Y[mask]
    if len(X) < 2:
        df['spread'] = np.nan
        return df, None
    if coefs is None:
        coefs = np.polyfit(X, Y, degree)
    Y_fit = np.polyval(coefs, X)
    spread = Y - Y_fit
    df['spread'] = np.nan
    df.loc[spread.index, 'spread'] = spread
    return df, coefs

# %%
combination = ('close_btc-usdt', 'close_eth-usdt')
spread_df, coefs = AddPolyfitSpread(prices_df, combination, coefs=None)
print(spread_df[['spread', combination[0], combination[1]]].head(10))

# %%
train_frac = 0.8
train_idx = int(train_frac * len(spread_df))
pre_data = spread_df.iloc[:train_idx].copy()
test_data = spread_df.iloc[train_idx:].copy()

# %%
# top_hmm = TopModelHMM(pre_data, window=24, reference_column='spread')
# preds = top_hmm.predict(test_data)
# print(preds[:10])

# %%
# top_arima = TopModelArima(pre_data, window=24, reference_column='spread')
# arima_preds = top_arima.predict(test_data)
# print(arima_preds[:10])

# %%
