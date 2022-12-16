import pandas as pd
import random


train_prop = 0.8

def main():
	data = pd.read_csv("processed/tweets_cn_news.csv")

	tweet_ids = data.tweetId.tolist()

	random.seed(0)
	random.shuffle(tweet_ids)

	train_ids = set(tweet_ids[:int(train_prop * len(tweet_ids))])
	rows = [{"tweetId": tweet_id, "split": "train" if tweet_id in train_ids else "val"} for tweet_id in tweet_ids]

	df = pd.DataFrame(rows)
	df.to_csv("processed/splits.csv", index=False)

if __name__ == "__main__":
	main()
