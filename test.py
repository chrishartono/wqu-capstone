#%%
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import test_topmodel

from backtester import Backtester
from startup_helpers import IsLoggingConfigured, ResetLogFileHandler, SetLogging
from top_model import TopModelType
from combinations import CreateAllPossibleCombinations
from comovement import ComovementType, test_cointegration
from feature_engineering import AddFeatures
from spread import AddPolyfitSpread
from target_creation import AddPeakNeighboursTarget, TargetType
from utils.helpers import CountAlternatingNonZeroSequences


def manual_test(prices_df: pd.DataFrame):
	train_frac = 0.8
	last_rows = 3000
	combination = ('close_sol-usdt', 'close_avax-usdt')

	# test_cointegration(prices_df[list(combination)], combination)

	pairs = set([c.split('_')[1] for c in combination])
	columns_to_choose = [col for col in prices_df.columns if col != 'date' and col.split('_')[1] in pairs]

	prices_df = prices_df[columns_to_choose]
	prices_df = prices_df.iloc[-last_rows:]

	train_split_idx = int(train_frac * len(prices_df))

	train = prices_df.iloc[:train_split_idx]
	test = prices_df.iloc[train_split_idx:]

	train, coefs = AddPolyfitSpread(train, combination, coefs=None)
	test, _ = AddPolyfitSpread(test, combination, coefs)

	train_days = (train.index[-1] - train.index[0]).days
	window_days = 10
	train, test = AddFeatures(train, test, combination, window_days)
	# feats_df = pd.concat([train, test], axis=0)
	# window_rows = int(len(train) / 20)
	# feats_df = AddPeakNeighboursSingleColumn(feats_df, combination, target_col='spread', period=window_rows, resulting_target_column='TARGET', numNeighbours=10)


def backtest_test(prices_df: pd.DataFrame, num_good_combs_to_choose: int, min_val_net_return: float, min_val_num_trades: int, use_top_model=TopModelType.HMM):

	all_possible_combinations = CreateAllPossibleCombinations(prices_df)
	# np.random.shuffle(all_possible_combinations)
	# all_possible_combinations_slice = all_possible_combinations[:50]

	# all_possible_combinations_slice = [('close_axs-usdt', 'close_sand-usdt')]
	# all_possible_combinations_slice = [('close_algo-usdt', 'close_reef-usdt')]
	# all_possible_combinations_slice = [('close_bat-usdt', 'close_omg-usdt')]
	# all_possible_combinations_slice = [('close_powr-usdt', 'close_algo-usdt'), ('close_troy-usdt', 'close_ach-usdt'), ('close_amp-usdt', 'close_clv-usdt'),
	# 								   ('close_rei-usdt', 'close_algo-usdt'), ('close_voxel-usdt', 'close_algo-usdt'), ('close_amp-usdt', 'close_bico-usdt'),
	# 								   ('close_badger-usdt', 'close_ach-usdt'), ('close_amp-usdt', 'close_celo-usdt'), ('close_rei-usdt', 'close_ach-usdt')]
	train_window_days = 660
	val_window_days = 60
	trade_window_days = 60
	spread_window = 5
	target_window = 20
	# train_window_days = (prices_df.index[-1] - prices_df.index[0]).days - trade_window_days
	# target_params = {'numNeighbours': 10, 'rolling_window_days': 10}
	target_params = {'look_ahead_days': target_window, 'reg_points_thresh_frac': 0.75, 'exceedance_thresh_frac': 0.001}
	backtester = Backtester(prices_df=prices_df,
							train_window_days=train_window_days,
							ml_val_window_days=trade_window_days,
							trade_window_days=trade_window_days,
							val_window_days=val_window_days,
							features_rolling_windows_days_list=[1, 5, 10],
							all_possible_combinations=all_possible_combinations,
							comovement_detection_type=ComovementType.COINT_MI,
							combination_limit=1000,
							trade_limit=1000,
							risk_free_rate=0,
							fees=0.1 / 100,
							min_val_net_return=min_val_net_return,
							min_val_num_trades=min_val_num_trades,
							num_good_combs_to_choose=num_good_combs_to_choose,
							use_top_model=None,
							target_type=TargetType.OLS_CLF,
							target_params=target_params,
							close_on_no_signal=False,
							spread_window=spread_window,
							use_parallelization=False,
							use_gpu=True)
	backtester.Run()

def run_consecutive_backtests():
	num_good_combs_to_choose_list = [1]
	min_val_net_return_list = [0.3]
	min_val_num_trades_list = [60]

	prices_df = pd.read_csv('dataset/binance_1h_ohlcv_2021-2025.csv', index_col='date', parse_dates=True)
	prices_df = prices_df[(prices_df.index >= '2022-01-01') & (prices_df.index <= '2025-03-01')]

	for num_good_combs_to_choose in num_good_combs_to_choose_list:
		for min_val_net_return in min_val_net_return_list:
			for min_val_num_trades in min_val_num_trades_list:
					now_str = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
					min_val_net_return_str = str(int(min_val_net_return*100))
					log_file_name = (f'logs/backtest_{now_str}_ncombs{num_good_combs_to_choose}_'
									 f'minret{min_val_net_return_str}_'
									 f'mintrd{min_val_num_trades}.log')

					if not IsLoggingConfigured():
						SetLogging(log_file_name)
					else:
						ResetLogFileHandler(log_file_name)

					prices_df_copy = prices_df.copy()
					backtest_test(prices_df_copy, num_good_combs_to_choose, min_val_net_return, min_val_num_trades)
					logging.info('Finished')

def ml_quality_test(prices_df: pd.DataFrame, train_window_days, val_window_days, trade_window_days, num_good_combs_to_choose, desired_num_samples,
					target_window, use_jump_features, use_copula_features, spread_window, filename, use_slice):
	all_possible_combinations = CreateAllPossibleCombinations(prices_df)

	# np.random.shuffle(all_possible_combinations)
	all_possible_combinations_slice = all_possible_combinations[:100]

	combinations_to_use = all_possible_combinations_slice if use_slice else all_possible_combinations

	target_params = {'look_ahead_days': target_window, 'reg_points_thresh_frac': 0.75, 'exceedance_thresh_frac': 0.001}

	settings_dic = {'train_window_days': train_window_days, 'target_look_ahead_days': target_params['look_ahead_days'], 'spread_window': spread_window,
					'jumps': use_jump_features, 'copula': use_copula_features}
	columns = ['train_window_days', 'target_look_ahead_days', 'spread_window', 'jumps', 'copula', 'class', 'metric', 'median', 'std']
	backtester = Backtester(prices_df=prices_df,
							train_window_days=train_window_days,
							ml_val_window_days=val_window_days,
							trade_window_days=trade_window_days,
							val_window_days=val_window_days,
							features_rolling_windows_days_list=[1, 5, 10],
							all_possible_combinations=combinations_to_use,
							comovement_detection_type=ComovementType.COINT_MI,
							combination_limit=1000,
							trade_limit=1000,
							risk_free_rate=0,
							fees=0.1 / 100,
							min_val_net_return=0.1,
							min_val_num_trades=trade_window_days * 5,
							num_good_combs_to_choose=num_good_combs_to_choose,
							use_top_model=None,
							target_type=TargetType.OLS_CLF,
							target_params=target_params,
							close_on_no_signal=False,
							spread_window=spread_window,
							use_parallelization=True,
							use_gpu=False)

	all_aggr_values = backtester.MLPredictionQualityTest(desired_num_samples=desired_num_samples, filename=filename)
	settings_aggr_values = [settings_dic | aggr for aggr in all_aggr_values]
	settings_aggr_df = pd.DataFrame(settings_aggr_values)
	settings_aggr_df = settings_aggr_df[columns]

	aggr_metrics_filename = 'aggregated_metrics.csv'

	if not os.path.isfile(aggr_metrics_filename):
		settings_aggr_df.to_csv(aggr_metrics_filename, header=columns, index=False)
	else:  # else it exists so append without writing the header
		settings_aggr_df.to_csv(aggr_metrics_filename, mode='a', header=False, index=False)

def run_consecutive_ml_quality_tests():
	now_str = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')


	val_window_days = 60
	trade_window_days = 60
	num_good_combs_to_choose = 100
	desired_num_samples = 3
	use_slice = False

	train_window_days_list = [720]
	target_window_list = [20]
	use_jump_features_list = [False]
	use_copula_features_list = [False]
	spread_windows_list = [5]
	prices_df = pd.read_csv('dataset/binance_1h_ohlcv_2021-2025.csv', index_col='date', parse_dates=True)

	for spread_window in spread_windows_list:
		for target_window in target_window_list:
			for train_window_days in train_window_days_list:
				for use_jump_features in use_jump_features_list:
					for use_copula_features in use_copula_features_list:
						filename = (f'{now_str}_jumps{use_jump_features}_copula{use_copula_features}_trn{train_window_days}_trd{trade_window_days}_'
									f'ncombs{num_good_combs_to_choose}_dsmpl{desired_num_samples}_tarwin{target_window}_spreadwin{spread_window}')
						log_file_name = f'logs/ml_test_{filename}.log'
						if not IsLoggingConfigured():
							SetLogging(log_file_name)
						else:
							ResetLogFileHandler(log_file_name)

						prices_df_copy = prices_df.copy()
						ml_quality_test(prices_df_copy, train_window_days, val_window_days, trade_window_days, num_good_combs_to_choose, desired_num_samples,
										target_window, use_jump_features, use_copula_features, spread_window, filename, use_slice)
						logging.info('Finished')

#%%

if __name__ == '__main__':
	now_str = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
	os.makedirs('logs', exist_ok=True)
	# parallel_logging(f'logs/wqu_capstone_{now_str}.log')

	# run_consecutive_ml_quality_tests()
	# TODO: Test run
	# prices_df = prices_df[(prices_df.index >= '2023-02-01') & (prices_df.index <= '2024-07-01')]
	# prices_df = prices_df[(prices_df.index >= '2022-01-01') & (prices_df.index <= '2024-09-01')]
	prices_df = pd.read_parquet('dataset/binance_1h_ohlcv_2021-2025.parquet')
	prices_df = prices_df.set_index("date")

	prices_df = prices_df[(prices_df.index >= '2022-01-01') & (prices_df.index <= '2024-09-01')]

	# manual_test(prices_df)
	log_file_name = (f'logs/backtest_test_60trades_target20_500combs.log')
	SetLogging(log_file_name)
	# prices_df = pd.read_csv('dataset/binance_1h_ohlcv_2021-2025.csv', index_col='date', parse_dates=True)
	# prices_df = prices_df[(prices_df.index >= '2022-01-01') & (prices_df.index <= '2024-09-01')]
	backtest_test(prices_df, num_good_combs_to_choose=500, min_val_net_return=0.1, min_val_num_trades=60)
	logging.info('Finished')
	#run_consecutive_backtests()


# %%
if __name__ == '__main__':
    now_str = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs('logs', exist_ok=True)

    # Load and ensure datetime index
    prices_df = pd.read_parquet('dataset/binance_1h_ohlcv_2021-2025.parquet')
    prices_df['date'] = pd.to_datetime(prices_df['date'], errors='coerce')  # Ensure 'date' is datetime
    prices_df = prices_df.set_index("date")
    prices_df = prices_df.sort_index()  # Ensure index is sorted
    prices_df = prices_df.astype('float64')

    # Filter by date range
    #prices_df = prices_df[(prices_df.index >= '2022-01-01') & (prices_df.index <= '2024-09-01')]

    # Check for empty DataFrame after filtering
    if prices_df.empty:
        raise ValueError(
            f"Filtered prices_df is empty! Available date range: {prices_df.index.min()} to {prices_df.index.max()}"
        )

    log_file_name = (f'logs/backtest_test_60trades_target20_500combs.log')
    SetLogging(log_file_name)

    backtest_test(prices_df, num_good_combs_to_choose=500, min_val_net_return=0.1, min_val_num_trades=60)
    logging.info('Finished')
# %%
def backtest_test(prices_df: pd.DataFrame, num_good_combs_to_choose: int, min_val_net_return: float, min_val_num_trades: int, use_top_model=TopModelType.HMM):

    # Only this combination!
    all_possible_combinations = [('close_vtho-usdt', 'close_win-usdt')]

    train_window_days = 660
    val_window_days = 60
    trade_window_days = 60
    spread_window = 5
    target_window = 20
    target_params = {'look_ahead_days': target_window, 'reg_points_thresh_frac': 0.75, 'exceedance_thresh_frac': 0.001}

    backtester = Backtester(
        prices_df=prices_df,
        train_window_days=train_window_days,
        ml_val_window_days=trade_window_days,
        trade_window_days=trade_window_days,
        val_window_days=val_window_days,
        features_rolling_windows_days_list=[1, 5, 10],
        all_possible_combinations=all_possible_combinations,
        comovement_detection_type=ComovementType.COINT_MI,
        combination_limit=1000,
        trade_limit=1000,
        risk_free_rate=0,
        fees=0.1 / 100,
        min_val_net_return=min_val_net_return,
        min_val_num_trades=min_val_num_trades,
        num_good_combs_to_choose=num_good_combs_to_choose,
        use_top_model=None,
        target_type=TargetType.OLS_CLF,
        target_params=target_params,
        close_on_no_signal=False,
        spread_window=spread_window,
        use_parallelization=False,      # (set to False while debugging)
    )
    backtester.Run()

# %%
from top_model import TopModelType
from comovement import ComovementType
from target_creation import TargetType
from backtester import Backtester
import pandas as pd

def backtest_test(prices_df: pd.DataFrame, num_good_combs_to_choose: int, min_val_net_return: float, min_val_num_trades: int, use_top_model=TopModelType.HMM):
    #all_possible_combinations = [('close_vtho-usdt', 'close_win-usdt')]
    all_possible_combinations = CreateAllPossibleCombinations(prices_df)
    train_window_days = 660
    val_window_days = 60
    trade_window_days = 60
    spread_window = 5
    target_window = 20
    target_params = {'look_ahead_days': target_window, 'reg_points_thresh_frac': 0.75, 'exceedance_thresh_frac': 0.001}
    backtester = Backtester(
        prices_df=prices_df,
        train_window_days=train_window_days,
        ml_val_window_days=trade_window_days,
        trade_window_days=trade_window_days,
        val_window_days=val_window_days,
        features_rolling_windows_days_list=[1, 5, 10],
        all_possible_combinations=all_possible_combinations,
        comovement_detection_type=ComovementType.COINT_MI,
        combination_limit=1000,
        trade_limit=1000,
        risk_free_rate=0,
        fees=0.1 / 100,
        min_val_net_return=min_val_net_return,
        min_val_num_trades=min_val_num_trades,
        num_good_combs_to_choose=num_good_combs_to_choose,
        use_top_model=TopModelType.HMM,
        target_type=TargetType.OLS_CLF,
        target_params=target_params,
        close_on_no_signal=False,
        spread_window=spread_window,
        use_parallelization=False,      # Debugging: single process
        use_gpu=False                   # <-- Make sure you add this!
    )
    backtester.Run()

# Run the backtest for just this pair
backtest_test(
    prices_df,
    num_good_combs_to_choose=1,
    min_val_net_return=0.1,
    min_val_num_trades=10
)

# %%
pairs = [
    ('close_algo-usdt', 'close_sand-usdt'),
    ('close_audio-usdt', 'close_lrc-usdt'),
    ('close_avax-usdt', 'close_utk-usdt'),
    ('close_axs-usdt', 'close_enj-usdt'),
    ('close_bch-usdt', 'close_nuls-usdt'),
    ('close_btc-usdt', 'close_uni-usdt'),
    ('close_coti-usdt', 'close_grt-usdt'),
    ('close_dent-usdt', 'close_hot-usdt'),
    ('close_hive-usdt', 'close_icx-usdt'),
    ('close_iota-usdt', 'close_ksm-usdt'),
    ('close_mbl-usdt', 'close_sc-usdt'),
    ('close_trx-usdt', 'close_xlm-usdt'),
    ('close_vtho-usdt', 'close_zrx-usdt'),
    ('close_vtho-usdt', 'close_win-usdt')
]

def backtest_selected_pairs(prices_df: pd.DataFrame, pairs, num_good_combs_to_choose: int, min_val_net_return: float, min_val_num_trades: int, use_top_model=None):
    backtester = Backtester(
        prices_df=prices_df,
        train_window_days=660,
        ml_val_window_days=60,
        trade_window_days=60,
        val_window_days=60,
        features_rolling_windows_days_list=[1, 5, 10],
        all_possible_combinations=pairs,   # <--- Only these pairs!
        comovement_detection_type=ComovementType.COINT_MI,
        combination_limit=1000,
        trade_limit=1000,
        risk_free_rate=0,
        fees=0.1 / 100,
        min_val_net_return=min_val_net_return,
        min_val_num_trades=min_val_num_trades,
        num_good_combs_to_choose=len(pairs),  # or set as needed
        use_top_model=use_top_model,
        target_type=TargetType.OLS_CLF,
        target_params={'look_ahead_days': 20, 'reg_points_thresh_frac': 0.75, 'exceedance_thresh_frac': 0.001},
        close_on_no_signal=False,
        spread_window=5,
        use_parallelization=False,
        use_gpu=False
    )
    backtester.Run()

# %%
backtest_selected_pairs(
    prices_df,
    pairs=pairs,
    num_good_combs_to_choose=len(pairs),
    min_val_net_return=0.1,
    min_val_num_trades=10,
    use_top_model=TopModelType.HMM   # or TopModelType.HMM if you want HMM
)

# %%
