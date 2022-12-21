import pandas as pd
import numpy as np

from sklearn import svm
from sklearn import metrics

import os
from pathlib import Path
basepath = Path(__file__).parent


def main():
	xcols = ["verified", "followers_count", "following_count", "tweet_count", "listed_count", "created_at"]
	ycol = "label"

	data, data_train, data_val = load_data()

	model = svm.SVC(probability=True, class_weight="balanced")
	model.fit(data_train[xcols], data_train[ycol])

	preds = model.predict(data_val[xcols])
	evaluate(data_val[ycol], preds)

	predictions = generate_predictions(model, data, xcols)

	os.makedirs(basepath.joinpath("output/"), exist_ok=True)
	predictions.to_csv(basepath.joinpath("output/userscores.csv"), index=False)

def load_data():
	data = pd.read_csv(basepath.joinpath("../../data/combined/processed/users_cn_news.csv"))
	data = data[["tweetId", "verified", "followers_count", "following_count", "tweet_count", "listed_count", "created_at"]]

	labels_df = pd.read_csv(basepath.joinpath("../../data/combined/processed/tweets_cn_news.csv"))
	labels_df = labels_df[["tweetId", "misleading"]]
	labels_df.rename(columns={"misleading": "label"}, inplace=True)
	labels_df.label = labels_df.label.apply(lambda x: 1 if x else -1)

	data = pd.merge(data, labels_df, on="tweetId")

	splits = pd.read_csv(basepath.joinpath("../../data/combined/processed/splits.csv"))
	train_ids = set(splits[splits["split"] == "train"]["tweetId"].values)
	val_ids = set(splits[splits["split"] == "val"]["tweetId"].values)

	data_train = data[data["tweetId"].isin(train_ids)]
	data_val = data[data["tweetId"].isin(val_ids)]

	return data, data_train, data_val

def generate_predictions(model, data, xcols):
	preds = model.predict(data[xcols])

	pred_scores = model.predict_proba(data[xcols])
	pred_scores = pred_scores[np.arange(pred_scores.shape[0]), preds]
	pred_scores = pred_scores * preds

	predictions = pd.DataFrame({
		"tweetId": data.tweetId,
		"userlabel": preds,
		"userscore": pred_scores,
	})
	return predictions

def evaluate(target, preds):
	report = metrics.classification_report(target, preds, target_names=["Not Misleading", "Misleading"])
	print(report)

	acc = metrics.accuracy_score(target, preds)
	precision = metrics.precision_score(target, preds)
	recall = metrics.recall_score(target, preds)

	print(f"User score accuracy:  {acc:.4f}")
	print(f"User score precision: {precision:.4f}")
	print(f"User score recall:    {recall:.4f}")


if __name__ == "__main__":
	main()
