import pandas as pd
import torch
import tqdm
from pathlib import Path
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from datasets import Dataset


# parameters

label_type = "emotion" # "sentiment"

basepath = Path("../")

# data

labels_df = pd.read_csv(basepath.joinpath("data/community-notes/processed/community_notes.csv"))
tweets_df = pd.read_csv(basepath.joinpath("data/community-notes/processed/tweets.csv"))
news_tweets_df = pd.read_csv(basepath.joinpath("data/news-tweets/processed/news_tweets.csv"))

tweets_df = tweets_df.loc[tweets_df.content.notna()]
news_tweets_df = news_tweets_df.loc[news_tweets_df.content.notna()]

labels_dict = labels_df.set_index("tweetId").to_dict("index")
tweets_dict = tweets_df.set_index("tweetId").to_dict("index")
news_tweets_dict = news_tweets_df.set_index("tweetId").to_dict("index")

cn_tweets_ids = list(set.intersection(set(labels_df.tweetId.tolist()), set(tweets_df.tweetId.tolist())))
news_tweets_ids = news_tweets_dict.keys()
tweet_ids = list(set.union(set(cn_tweets_ids), set(news_tweets_ids)))

cn_data = [{"id": i, "text": tweets_dict[i]["content"], "label": int(labels_dict[i]["misleading"])} for i in cn_tweets_ids]
news_data = [{"id": i, "text": news_tweets_dict[i]["content"], "label": 0} for i in news_tweets_ids]

alldata = cn_data + news_data
dataset_all = Dataset.from_list(alldata)

# model setup

if label_type == "sentiment":
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
elif label_type == "emotion":
    MODEL = "cardiffnlp/twitter-roberta-base-emotion"
else:
    raise ValueError("Invalid label type")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

inf_pipeline = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=torch.device("cuda:0"))

# inference

results = {ex["id"]: inf_pipeline(ex["text"]) for ex in tqdm.tqdm(dataset_all)}

# output

results_list = [{"tweetId": i, **results.get(i)[0]} for i in tweet_ids if results.get(i) is not None]
results_df = pd.DataFrame(results_list)

os.makedirs(basepath.joinpath("results/"), exist_ok=True)
results_df.to_csv(basepath.joinpath(f"results/tweets-{label_type}.csv"), index=False)
