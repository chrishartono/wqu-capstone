import logging
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from utils.helpers import DaysWindowToPeriods

catboost_hyperparameters = {'depth' : 3, 'iterations': 1000, 'loss_function': 'MultiClass', 'learning_rate': 0.1}

def show_clf_results(y_test: pd.Series, y_probs: np.ndarray, y_pred: np.ndarray):
	micro_roc_auc_ovr = roc_auc_score(y_test, y_probs, multi_class="ovr", average="micro")
	logging.info(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")

	report = classification_report(y_test, y_pred)
	logging.info(f"Classification report:\n{report}")

	fpr, tpr, _ = roc_curve(y_test, y_probs)
	plt.plot(fpr,
			 tpr,
			label=f"micro-average ROC curve (AUC = {micro_roc_auc_ovr:.2f})",
			color="deeppink",
			linestyle=":",
			linewidth=4)
	plt.show()


def Train(train: pd.DataFrame, test: pd.DataFrame, val_window_days: int):
	val_window_periods = DaysWindowToPeriods(train, val_window_days)

	val = train.iloc[-val_window_periods:]
	train = train.iloc[:len(train) - val_window_periods]

	X_train = train.drop(columns=['TARGET'])
	X_val = val.drop(columns=['TARGET'])
	X_test = test.drop(columns=['TARGET'])
	y_train = train['TARGET']
	y_val = val['TARGET']
	y_test = test['TARGET']

	clf = CatBoostClassifier(verbose=0, **catboost_hyperparameters)
	clf.fit(X=X_train, y=y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)

	logging.info(f'Making predictions for {len(X_test)} rows')
	y_probs = clf.predict_proba(X_test)
	y_pred = clf.predict(X_test)

	show_clf_results(y_test, y_probs, y_pred)

	return y_pred
