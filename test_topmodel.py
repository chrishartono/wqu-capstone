#%%
#test 
from hmmlearn import hmm
from top_model import TopModelHMM,TopModelArima
from spread import AddPolyfitSpread
import pandas as pd
import numpy as np

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

