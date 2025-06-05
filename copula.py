from datetime import datetime
from statsmodels.regression.rolling import RollingOLS
import pandas as pd
import numpy as np
from statsmodels.distributions.copula.api import GaussianCopula, StudentTCopula, GumbelCopula, FrankCopula, ClaytonCopula
from scipy.stats import rankdata, gaussian_kde
from scipy.stats import norm as _norm, t as _t


def calculate_spread(reference_price, asset_price, window=20):
	"""
	Calculate the spread between a reference price and an asset price using a rolling beta.
	The spread is calculated as:
	spread = reference_price - beta * asset_price

	where beta is the rolling regression coefficient of asset_price on reference_price.
	"""
	# rolling beta using RollingOLS
	asset_price_adj = asset_price.iloc[1:].reset_index(drop=True)
	reference_price_adj = reference_price.iloc[1:].reset_index(drop=True)
	asset_price_adj.name = asset_price.name
	model_rols = RollingOLS(endog=reference_price_adj,
							exog=asset_price_adj,
							window=window,
							min_nobs=window)

	results_rols = model_rols.fit()
	rolling_beta_values_adj = results_rols.params[asset_price_adj.name]
	beta = pd.Series(index=asset_price.index, dtype=float)
	valid_betas = rolling_beta_values_adj.dropna().values
	if len(valid_betas) > 0:
		beta.iloc[window: window + len(valid_betas)] = valid_betas

	# calculate spread
	spread = reference_price - beta * asset_price
	return spread

def ecdf_transform(series):
	ranks = rankdata(series, method='average')
	return ranks / (len(series) + 1)  # (n+1) to avoid 0 and 1


def copula_cond_prob(u, v, rho=None, df=None):
	"""
	P(U ≤ u | V = v) for Gaussian, StudentT, Gumbel, Clayton, or Frank copula.
	"""
	x = _norm.ppf(u)
	y = _norm.ppf(v)

	# t-Student copula
	mu = rho * y
	sigma = np.sqrt((df + y ** 2) * (1 - rho ** 2) / (df + 1))
	t_val = (x - mu) / sigma

	return _t.cdf(t_val, df + 1)


def calc_signals_article(
		cond_df: pd.DataFrame,
		alpha1: float = 0.10,
		alpha2: float = 0.10
		) -> np.ndarray:
	"""
	Parameters
	----------
	alpha1 : float, default 0.10
		Порог для входа в позиции
	alpha2 : float, default 0.10
		Допуск вокруг 0.5 для выхода/нейтрализации.

	Returns
	-------
	np.ndarray[int]
		Сигналы:  1 — long S1 / short S2,
				 -1 — short S1 / long S2,
				  0 — hold или закрыть.
	"""
	df = cond_df
	h12 = df["h12"].values
	h21 = df["h21"].values

	signals = np.zeros(len(df), dtype=int)

	# --- вход в long S1 / short S2 ---
	long_mask = (h12 < alpha1) & (h21 > 1 - alpha1)
	signals[long_mask] = 1

	# --- вход в short S1 / long S2 ---
	short_mask = (h12 > 1 - alpha1) & (h21 < alpha1)
	signals[short_mask] = -1

	# --- выход / нейтральная зона (оставляем 0) ---
	close_mask = (np.abs(h12 - 0.5) < alpha2) & (np.abs(h21 - 0.5) < alpha2)
	signals[close_mask] = 0  # для ясности; по умолчанию уже 0

	return signals


def calc_signals_rolling(cond_df, win_mean=20, alpha1=0.3, alpha2=0.0):
	h12 = cond_df['h12'].rolling(win_mean, center=False).mean().fillna(0.5)
	h21 = cond_df['h21'].rolling(win_mean, center=False).mean().fillna(0.5)
	cond_df_article = pd.DataFrame({'h12': h12, 'h21': h21})
	sig_article = calc_signals_article(cond_df_article, alpha1=alpha1, alpha2=alpha2)
	# sig_article_input = np.concatenate((np.zeros(window_spread - 1), sig_article))

	return sig_article

def make_spreads(feats_df: pd.DataFrame, combination: tuple[str, str], reference_prices: pd.DataFrame, window: int):
	spreads = {}
	for pair_col in combination:
		spread = calculate_spread(reference_prices, feats_df[pair_col], window)
		spreads[f"{pair_col}"] = spread
	spreads_df = pd.DataFrame(spreads)

	return spreads_df

def make_signal(combination: tuple[str, str], U: pd.DataFrame, rho_uv: np.ndarray, df_hat: int, spread_window: int):
	u, v = U[combination[0]], U[combination[1]]
	cond_u = copula_cond_prob(u, v, rho=rho_uv, df=df_hat)
	cond_v = copula_cond_prob(v, u,  rho=rho_uv, df=df_hat)

	cond_df = pd.DataFrame({'h12': cond_u, 'h21': cond_v})
	signal = calc_signals_article(cond_df, alpha1=0.3, alpha2=0.3)

	# Add zeros at the beginning to match the original size
	signal = np.pad(signal, (spread_window, 0), 'constant', constant_values=0)
	return signal

def make_U(spreads_df: pd.DataFrame):
	cdf_input = spreads_df.dropna()
	U = cdf_input.apply(ecdf_transform)

	return U

def do_fit_copula(u_data: np.ndarray):
	rho_uv = StudentTCopula().fit_corr_param(u_data)
	corr_matrix = np.array([[1, rho_uv], [rho_uv, 1]])
	grid = np.arange(2, 30)
	ll = [StudentTCopula(corr=corr_matrix, df=df_i).logpdf(u_data).sum() for df_i in grid]
	df_hat = grid[np.argmax(ll)]

	return rho_uv, df_hat

def FitCopula(train: pd.DataFrame, combination: tuple[str, str], train_reference_prices: pd.DataFrame, spread_window: int):
	spreads_df = make_spreads(train, combination, train_reference_prices.iloc[:,0], spread_window)

	U = make_U(spreads_df)
	u_data = U.to_numpy()

	rho_uv, df_hat = do_fit_copula(u_data)
	signal = make_signal(combination, U, rho_uv, df_hat, spread_window)

	return rho_uv, df_hat, signal

def CalcCopulaSignals(test: pd.DataFrame,
					  combination: tuple[str, str],
					  test_reference_prices: pd.DataFrame,
					  spread_window: int,
					  rho_uv: np.ndarray,
					  df_hat: int):
	spreads_df = make_spreads(test, combination, test_reference_prices.iloc[:,0], spread_window)
	U = make_U(spreads_df)
	signal = make_signal(combination, U, rho_uv, df_hat, spread_window)

	return signal
