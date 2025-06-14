import logging
import sys
import warnings

from threshold_tuner import ClassificationThresholdTuner

warnings.filterwarnings("ignore")
from itertools import cycle
from tqdm import tqdm
from enum import IntEnum

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import (PrecisionRecallDisplay, average_precision_score, classification_report, f1_score, precision_recall_curve, roc_auc_score, roc_curve,
							 auc, )
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import compute_class_weight
from statsmodels.tsa.arima.model import ARIMA

from top_model import TopModelArima, TopModelType
from utils.helpers import CountAlternatingNonZeroSequences, DaysWindowToPeriods, LogValueCounts

catboost_hyperparameters = {
		'depth'        : 5,
		'iterations'   : 1000, # 100
		'loss_function': 'MultiClass',
		'learning_rate': 0.01, # 0.1
		'random_state' : 13579
		# 'rsm': 0.8,
		# 'reg_lambda': 0.5
		# 'thread_count': 1
		}


def calc_multiclass_macro_auc(y_train: pd.Series, y_test: pd.Series, y_probs: np.ndarray):
	label_binarizer = LabelBinarizer().fit(y_train)
	y_onehot_test = label_binarizer.transform(y_test)

	n_classes = len(np.unique(y_test))
	fpr_list = []
	tpr_list = []
	for i in range(n_classes):
		fpr_class, tpr_class, _ = roc_curve(y_onehot_test[:, i], y_probs[:, i])
		fpr_list.append(fpr_class)
		tpr_list.append(tpr_class)

	fpr_grid = np.linspace(0.0, 1.0, 1000)

	# Interpolate all ROC curves at these points
	mean_tpr = np.zeros_like(fpr_grid)

	for i in range(n_classes):
		mean_tpr += np.interp(fpr_grid, fpr_list[i], tpr_list[i])  # linear interpolation

	# Average it and compute AUC
	mean_tpr /= n_classes

	fpr = fpr_grid
	tpr = mean_tpr

	return fpr, tpr, fpr_list, tpr_list


def save_roc_plot(combination: tuple[str, str], y_train: pd.Series, y_test: pd.Series, y_probs: np.ndarray):
	macro_roc_auc_ovr = roc_auc_score(y_test, y_probs, multi_class="ovr", average="macro")
	logging.info(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")

	fpr, tpr, fpr_classes_list, tpr_classes_list = calc_multiclass_macro_auc(y_train, y_test, y_probs)
	fig, ax = plt.subplots(nrows=1, ncols=1)

	# colors = ["aqua", "darkorange", "cornflowerblue"]
	for i in range(len(fpr_classes_list)):
		fpr_class = fpr_classes_list[i]
		tpr_class = tpr_classes_list[i]
		auc_class = auc(fpr_class, tpr_class)
		ax.plot(fpr_class, tpr_class, label=f"ROC curve for class {i} (AUC = {auc_class:.2f})")

	# RocCurveDisplay.from_predictions(
	# 		y_onehot_test[:, class_id],
	# 		y_score[:, class_id],
	# 		name=f"ROC curve for {target_names[class_id]}",
	# 		color=color,
	# 		ax=ax,
	# 		plot_chance_level=(class_id == 2),
	# 		despine=True,
	# 		)

	ax.plot(fpr,
			tpr,
			label=f"Macro-average ROC curve (AUC = {macro_roc_auc_ovr:.2f})",
			color="deeppink",
			linestyle=":",
			linewidth=4)
	_ = ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"{combination} Macro-average ROC curve")
	ax.legend()
	fig.savefig(f'auc.png')
	plt.show()


def save_pr_plot(combination: tuple[str, str], y_train: pd.Series, y_test: pd.Series, y_probs: np.ndarray):
	label_binarizer = LabelBinarizer().fit(y_train)
	y_onehot_test = label_binarizer.transform(y_test)
	n_classes = len(np.unique(y_test))

	# For each class
	precision = dict()
	recall = dict()
	average_precision = dict()
	for i in range(n_classes):
		precision[i], recall[i], _ = precision_recall_curve(y_onehot_test[:, i], y_probs[:, i])
		average_precision[i] = average_precision_score(y_onehot_test[:, i], y_probs[:, i])

	# A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"], _ = precision_recall_curve(y_onehot_test.ravel(), y_probs.ravel())
	average_precision["micro"] = average_precision_score(y_onehot_test, y_probs, average="micro")

	# setup plot details
	# colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

	fig, ax = plt.subplots(figsize=(9, 9))
	#
	# f_scores = np.linspace(0.2, 0.8, num=4)
	# lines, labels = [], []
	# for f_score in f_scores:
	# 	x = np.linspace(0.01, 1)
	# 	y = f_score * x / (2 * x - f_score)
	# 	(l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
	# 	plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

	display = PrecisionRecallDisplay(recall=recall["micro"], precision=precision["micro"], average_precision=average_precision["micro"])
	display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

	# for i, color in zip(range(n_classes), colors):
	# 	display = PrecisionRecallDisplay(recall=recall[i], precision=precision[i], average_precision=average_precision[i])
	# 	display.plot( ax=ax, name=f"Precision-recall for class {i}", color=color)

	for i in range(n_classes):
		display = PrecisionRecallDisplay(recall=recall[i], precision=precision[i], average_precision=average_precision[i])
		display.plot(ax=ax, name=f"Precision-recall for class {i}")

	# add the legend for the iso-f1 curves
	# handles, labels = display.ax_.get_legend_handles_labels()
	# handles.extend([l])
	# labels.extend(["iso-f1 curves"])
	# set the legend and the axes
	# ax.legend(handles=handles, labels=labels, loc="best")
	ax.set_title(f"{combination} Precision-Recall curves")
	ax.set_ylim([0, 1.2])
	ax.legend(loc="best")
	fig.savefig(f'pr-re.png')
	plt.show()


def save_feature_importance(combination: tuple[str, str], clf: CatBoostClassifier, columns: list[str]):
	feature_importance = clf.feature_importances_
	sorted_idx = np.argsort(feature_importance)[-30:]

	fig = plt.figure(figsize=(12, 6))
	plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
	plt.yticks(range(len(sorted_idx)), np.array(columns)[sorted_idx])
	plt.title(f'{combination} Feature Importance')
	plt.tight_layout()
	fig.savefig(f'feature_importances.png')

	plt.show()


def save_clf_results(combination: tuple[str, str],
					 clf: CatBoostClassifier,
					 columns: list[str],
					 y_train: pd.Series,
					 y_test: pd.Series,
					 y_probs: np.ndarray,
					 y_pred: np.ndarray):
	LogValueCounts(y_test.unique(), y_test.value_counts(sort=False).values, 'Test', len(y_test))

	report = classification_report(y_test, y_pred)
	logging.info(f"Classification report:\n{report}")

	save_roc_plot(combination, y_train, y_test, y_probs)
	save_pr_plot(combination, y_train, y_test, y_probs)
	save_feature_importance(combination, clf, columns)


def Train(train: pd.DataFrame, test: pd.DataFrame, combination: tuple[str, str], val_window_days: int, categorical_features: list[str], use_gpu: bool):
	logging.info(f'Start bottom model training for {combination}')

	val_window_periods = DaysWindowToPeriods(train, val_window_days)

	val = train.iloc[-val_window_periods:]
	train = train.iloc[:len(train) - val_window_periods]
	
	# Exclude non-scaled close price
	# close_columns = [col for col in train.columns if col.__contains__('close') and col.count('_') == 1]
	# X_train = train.drop(columns=['TARGET'] + close_columns)
	# X_test = test.drop(columns=['TARGET'] + close_columns)
	X_train = train.drop(columns=['TARGET'])
	X_val = val.drop(columns=['TARGET'])
	X_test = test.drop(columns=['TARGET'])
	y_train = train['TARGET']
	y_val = val['TARGET']
	y_test = test['TARGET']

	# sys.exit(0)

	LogValueCounts(y_train.unique(), y_train.value_counts(sort=False).values, 'Train', len(y_train))

	if use_gpu:
		catboost_hyperparameters['task_type'] = 'GPU'
		catboost_hyperparameters['devices'] = '0'

	# clf = CatBoostClassifier(verbose=0, class_weights=class_weights, **catboost_hyperparameters)
	clf = CatBoostClassifier(verbose=0, cat_features=categorical_features, **catboost_hyperparameters)
	clf.fit(X=X_train, y=y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
	# clf.fit(X=X_train, y=y_train)

	logging.info(f'Making predictions for {len(X_test)} rows')
	y_pred = clf.predict(X_test)

	train_signal_density = CountAlternatingNonZeroSequences(y_train) / (len(y_train) / 24)
	test_signal_density = CountAlternatingNonZeroSequences(y_test) / (len(y_test) / 24)
	pred_signal_density = CountAlternatingNonZeroSequences(y_pred) / (len(y_pred) / 24)

	logging.info(f'{train_signal_density=:.3f} {test_signal_density=:.3f} {pred_signal_density=:.3f}')

	# y_probs = clf.predict_proba(X_test)
	# save_clf_results(combination, clf, list(X_train.columns), y_train, y_test, y_probs, y_pred)
	# del y_probs
	# sys.exit(0)

	del train, X_train, X_test, y_train, y_test
	# del val, train, X_train, X_val, X_test, y_train, y_val, y_test

	return y_pred, clf


def ResearchTrain(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, combination: tuple[str, str], categorical_features: list[str], use_gpu: bool):
	logging.info(f'Start bottom model training for {combination}')

	X_train = train.drop(columns=['TARGET'])
	X_val = val.drop(columns=['TARGET'])
	X_test = test.drop(columns=['TARGET'])
	y_train = train['TARGET']
	y_val = val['TARGET']
	y_test = test['TARGET']

	tuner = ClassificationThresholdTuner()

	LogValueCounts(y_train.unique(), y_train.value_counts(sort=False).values, 'Train', len(y_train))

	if use_gpu:
		catboost_hyperparameters['task_type'] = 'GPU'
		catboost_hyperparameters['devices'] = '0'

	clf = CatBoostClassifier(verbose=0, cat_features=categorical_features, **catboost_hyperparameters)
	# clf.fit(X=X_train, y=y_train)
	clf.fit(X=X_train, y=y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
	# y_pred = clf.predict(X_test)
	y_val_probs = clf.predict_proba(X_val)
	y_test_probs = clf.predict_proba(X_test)
	y_test_pred_orig = clf.predict(X_test)

	target_classes = sorted(np.unique(y_val))
	best_thresholds = tuner.tune_threshold(y_true=y_val,
										   target_classes=target_classes,
										   y_pred_proba=y_val_probs,
										   metric=f1_score,
										   average='macro',
										   higher_is_better=True,
										   default_class='0',
										   max_iterations=5)

	# tuner.print_stats_proba(y_true=y_val,
	# 						target_classes=target_classes,
	# 						y_pred_proba=y_val_probs,
	# 						default_class='0',
	# 						thresholds=[0.5, 0.5, 0.5])
	#
	# tuner.print_stats_proba(y_true=y_val,
	# 						target_classes=target_classes,
	# 						y_pred_proba=y_val_probs,
	# 						default_class='0',
	# 						thresholds=best_thresholds)

	# orig_report = classification_report(y_test, y_test_pred_orig)
	# logging.info(f"Original Classification report:\n{orig_report}")

	y_test_pred_tuned = tuner.get_predictions(target_classes, y_test_probs, '0', best_thresholds)
	y_test_pred_tuned = [int(label) for label in y_test_pred_tuned]
	# tuned_report = classification_report(y_test, y_test_pred_tuned)
	# logging.info(f"TUNED Classification report:\n{tuned_report}")

	classes_default_f1_scores = f1_score(y_test, y_test_pred_orig, average=None)
	macro_default_f1_score = f1_score(y_test, y_test_pred_orig, average='macro')

	classes_tuned_f1_scores = f1_score(y_test, y_test_pred_tuned, average=None)
	macro_tuned_f1_score = f1_score(y_test, y_test_pred_tuned, average='macro')

	macro_roc_auc_ovr = roc_auc_score(y_test, y_test_probs, multi_class="ovr", average="macro")
	metrics = {'auc_macro': macro_roc_auc_ovr, 'f1_default_macro': macro_default_f1_score, 'f1_tuned_macro': macro_tuned_f1_score}

	fpr, tpr, fpr_classes_list, tpr_classes_list = calc_multiclass_macro_auc(y_train, y_test, y_test_probs)
	for i in range(len(fpr_classes_list)):
		fpr_class = fpr_classes_list[i]
		tpr_class = tpr_classes_list[i]
		auc_class = auc(fpr_class, tpr_class)

		metrics[f'auc_{i}'] = auc_class
		metrics[f'f1_default_{i}'] = classes_default_f1_scores[i]
		metrics[f'f1_tuned_{i}'] = classes_tuned_f1_scores[i]

	return metrics


def apply_top_model_filter(y_signal, y_filter):
	y_signal_filtered = y_signal.reshape(-1) * np.array(y_filter)
	y_pred = y_signal_filtered.reshape(-1, 1)
	return y_pred


def Predict(data: pd.DataFrame,
			data_val: pd.DataFrame,
			model, combination: tuple[str, str],
			use_top_model: TopModelType):
	logging.info(f'Start bottom model training for {combination}')
	# Exclude non-scaled close price
	# close_columns = [col for col in data.columns if col.__contains__('close') and col.count('_') == 1]
	# X = data.drop(columns=['TARGET']+close_columns)
	X = data.drop(columns=['TARGET'])
	y_pred = model.predict(X)

	if use_top_model == TopModelType.ARIMA:
		top_model_arima = TopModelArima(pre_data=data_val, window=24, reference_column='spread')
		y_pred_top_model = top_model_arima.predict(data)

		y_pred = apply_top_model_filter(y_signal=y_pred, y_filter=y_pred_top_model)
	elif use_top_model == TopModelType.HMM:
		# TODO
		pass

	return y_pred
