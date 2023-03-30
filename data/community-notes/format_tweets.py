import pandas as pd
import json
import os
from cleantext import clean


with open("downloads/tweets.json", "r") as f:
	twitter_raw = json.load(f)

tweets_raw = twitter_raw["data"]
tweets_raw = {tweet["id"]: tweet for tweet in tweets_raw}

users_raw = twitter_raw["includes"]["users"]
users = {user["id"]: user for user in users_raw}

def process_content(tweet):
	text = tweet["text"]
	text = clean_text(text)
	return text

def clean_text(text):
	return clean(
		text,
		fix_unicode=True,
		to_ascii=True,
		lower=False,
		no_line_breaks=True,
		no_urls=True, replace_with_url="",
		no_emoji=True,
		no_punct=False,
		lang="en",
	)

def format_tweet(tweet_id):
	tweet = tweets_raw[tweet_id]
	content = process_content(tweet)

	output = {
		"tweetId": tweet_id,
		"content": content,

		"created_at": tweet["created_at"],

		"conversation_id": tweet["conversation_id"],

		"userId": tweet["author_id"],
		"user_name": users[tweet["author_id"]]["name"],
		"user_username": users[tweet["author_id"]]["username"],
		"user_verified": users[tweet["author_id"]]["verified"],
		"user_img": users[tweet["author_id"]]["profile_image_url"],
		
		"likes": tweet["public_metrics"]["like_count"],
		"retweets": tweet["public_metrics"]["retweet_count"],
		"replies": tweet["public_metrics"]["reply_count"],
		"quotes": tweet["public_metrics"]["quote_count"],
	}
	return output

tweet_ids = tweets_raw.keys()
tweets = [format_tweet(tweet_id) for tweet_id in tweet_ids]

tweets_df = pd.DataFrame(tweets)

os.makedirs("processed/", exist_ok=True)
tweets_df.to_csv("processed/community_notes_tweets.csv", index=False)
