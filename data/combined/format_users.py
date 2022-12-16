import pandas as pd
import json
import os
import datetime


with open("downloads/users.json", "r") as f:
	twitter_raw = json.load(f)

users_raw = twitter_raw["data"]
users_raw = {user["id"]: user for user in users_raw}

def format_user(user_id):
	user = users_raw[user_id]

	ts = datetime.datetime.strptime(user["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
	ts = int(datetime.datetime.timestamp(ts))

	output = {
		"userId": user_id,

		"verified": int(user["verified"]),
		"created_at": ts,
		
		"followers_count": user["public_metrics"]["followers_count"],
		"following_count": user["public_metrics"]["following_count"],
		"tweet_count": user["public_metrics"]["tweet_count"],
		"listed_count": user["public_metrics"]["listed_count"],
	}
	return output

with open("downloads/user-errors.json", "r") as f:
	err_ids = json.load(f)

user_ids = users_raw.keys()
users = [format_user(user_id) for user_id in user_ids]
for i in err_ids:
	users.append({
		"userId": i,
		"verified": 0,
		"created_at": int(datetime.datetime.timestamp(datetime.datetime.now())),
		"followers_count": 0,
		"following_count": 0,
		"tweet_count": 0,
		"listed_count": 0,
	})
users_df = pd.DataFrame(users)

tweets_df = pd.read_csv("processed/tweets_cn_news.csv", dtype={"tweetId": str, "userId": str})
tweets_df = tweets_df[["tweetId", "userId"]]

users_df = users_df.merge(tweets_df, on="userId", how="inner")

os.makedirs("processed/", exist_ok=True)
users_df.to_csv("processed/users_cn_news.csv", index=False)
