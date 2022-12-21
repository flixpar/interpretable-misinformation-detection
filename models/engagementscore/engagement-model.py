import pandas as pd
import numpy as np

from sklearn import svm
from sklearn import metrics

import os
from pathlib import Path
basepath = Path(__file__).parent


def main():
	xcols = ["likes", "replies", "retweets", "quotes"]
	ycol = "label"

	data, data_train, data_val = load_data()

	model = svm.SVC(probability=True)
	model.fit(data_train[xcols], data_train[ycol])

	preds = model.predict(data_val[xcols])
	evaluate(data_val[ycol], preds)

	predictions = generate_predictions(model, data, xcols)

	os.makedirs(basepath.joinpath("output/"), exist_ok=True)
	predictions.to_csv(basepath.joinpath("output/engagementscores.csv"), index=False)

def load_data():
	data = pd.read_csv(basepath.joinpath("../../data/combined/processed/tweets_cn_news.csv"))
	data = data[["tweetId", "misleading", "likes", "replies", "retweets", "quotes"]]

	data.rename(columns={"misleading": "label"}, inplace=True)
	data.label = data.label.apply(lambda x: 1 if x else -1)

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
		"engagementlabel": preds,
		"engagementscore": pred_scores,
	})
	return predictions

def evaluate(target, preds):
	report = metrics.classification_report(target, preds, target_names=["Not Misleading", "Misleading"])
	print(report)

	acc = metrics.accuracy_score(target, preds)
	precision = metrics.precision_score(target, preds)
	recall = metrics.recall_score(target, preds)

	print(f"Engagement score accuracy:  {acc:.4f}")
	print(f"Engagement score precision: {precision:.4f}")
	print(f"Engagement score recall:    {recall:.4f}")


if __name__ == "__main__":
	main()
