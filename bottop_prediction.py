import logging
import sys
import warnings

warnings.filterwarnings("ignore")
from itertools import cycle
from tqdm import tqdm
from enum import IntEnum

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, classification_report, precision_recall_curve, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import compute_class_weight
from statsmodels.tsa.arima.model import ARIMA

from utils.helpers import DaysWindowToPeriods, LogValueCounts

catboost_hyperparameters = {'depth': 4, 'iterations': 1000, 'loss_function': 'MultiClass', 'learning_rate': 0.1}


class TopModelType(IntEnum):
	HMM = 0
	ARIMA = 1


class TopModelArima:
	def __init__(
			self,
			pre_data: pd.DataFrame,
			window: int = 24):

		self.__window = window
		self.__pre_data = pre_data

		self.__std_history = self.backfill_std_history(pre_data, window)

	def backfill_std_history(self, pre_data: pd.DataFrame, window: int):
		history = []

		for i in tqdm(range(window, len(pre_data)), desc='backfilling ARIMA std history'):
			model_ = ARIMA(pre_data.iloc[i - window:i]['spread'], order=(1, 0, 1))
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
						], axis=0)['spread']
			else:
				x = data.iloc[i - self.__window:i]['spread']

			model_ = ARIMA(x, order=(1, 0, 1))
			res = model_.fit()
			std = np.std(res.resid)

			threshold = np.quantile(self.__std_history, q=0.9)
			pred.append(1 if std > threshold else 0)

			self.__std_history.append(std)

		return pred


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

	save_pr_plot(combination, y_train, y_test, y_probs)
	save_feature_importance(combination, clf, columns)

def Train(train: pd.DataFrame, test: pd.DataFrame, combination: tuple[str, str], val_window_days: int):
	logging.info(f'Start bottom model training for {combination}')

	# val_window_periods = DaysWindowToPeriods(train, val_window_days)

	# val = train.iloc[-val_window_periods:]
	# train = train.iloc[:len(train) - val_window_periods]

	X_train = train.drop(columns=['TARGET'])
	# X_val = val.drop(columns=['TARGET'])
	X_test = test.drop(columns=['TARGET'])
	y_train = train['TARGET']
	# y_val = val['TARGET']
	y_test = test['TARGET']

	# classes = np.unique(y_train)
	# weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
	# class_weights = dict(zip(classes, weights))

	LogValueCounts(y_train.unique(), y_train.value_counts(sort=False).values, 'Train', len(y_train))

	# clf = CatBoostClassifier(verbose=0, class_weights=class_weights, **catboost_hyperparameters)
	clf = CatBoostClassifier(verbose=0, **catboost_hyperparameters)
	# clf.fit(X=X_train, y=y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
	clf.fit(X=X_train, y=y_train)

	logging.info(f'Making predictions for {len(X_test)} rows')
	y_probs = clf.predict_proba(X_test)
	y_pred = clf.predict(X_test)

	# save_clf_results(combination, clf, list(X_train.columns), y_train, y_test, y_probs, y_pred)

	del train, X_train, X_test, y_train, y_test, y_probs
	# del val, train, X_train, X_val, X_test, y_train, y_val, y_test, y_probs

	return y_pred, clf


def Predict(data: pd.DataFrame,
			data_val: pd.DataFrame,
			model, combination: tuple[str, str],
			use_top_model: TopModelType):
	logging.info(f'Start bottom model training for {combination}')
	X = data.drop(columns=['TARGET'])
	y_pred = model.predict(X)

	if use_top_model == TopModelType.ARIMA:
		top_model_arima = TopModelArima(pre_data=data_val, window=24)
		y_pred_top_model = top_model_arima.predict(data)

		y_pred_filtered = y_pred.reshape(-1) * np.array(y_pred_top_model)
		y_pred = y_pred_filtered.reshape(-1, 1)
	elif use_top_model == TopModelType.HMM:
		# TODO
		pass

	return y_pred
