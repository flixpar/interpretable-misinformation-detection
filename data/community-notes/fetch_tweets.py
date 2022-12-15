import pandas as pd
import requests
import os
import tqdm
import glob
import json


DEBUG = False

url = "https://api.twitter.com/2/tweets"

bearer_token = os.environ["TWITTER_BEARER_TOKEN"]
headers = {"Authorization": f"Bearer {bearer_token}"}

# payload_base = {
# 	"tweet.fields": "created_at,public_metrics",
# 	"expansions": "author_id",
# 	"user.fields": "name,username,verified,profile_image_url",
# }
payload_base = {
	"tweet.fields": "created_at,public_metrics,conversation_id,entities,context_annotations",
	"expansions": "author_id,entities.mentions.username,referenced_tweets.id",
	"user.fields": "name,username,verified,profile_image_url",
}

def fetch_tweets(tweet_ids):
	payload = payload_base.copy()
	payload["ids"] = ",".join(tweet_ids)
	response = requests.request("GET", url, headers=headers, data=payload)
	return response

def merge_outputs():
	fns = glob.glob("downloads/tweets/*.json")

	data = {"data": [], "includes": {"users": []}}
	errors = []

	for fn in fns:
		with open(fn, "r") as f:
			f_data = json.load(f)

			if "data" in f_data:
				data["data"].extend(f_data["data"])
				data["includes"]["users"].extend(f_data["includes"]["users"])
			else:
				print("No data in:", fn)

			if "errors" in f_data:
				errors.extend([e["resource_id"] for e in f_data["errors"]])

	with open("downloads/tweets.json", "w") as f:
		json.dump(data, f, indent="\t")

	with open("downloads/tweet-errors.json", "w") as f:
		json.dump(errors, f, indent="\t")

def get_tweet_ids():
	rawdata = pd.read_csv("processed/community_notes.csv", dtype={"tweetId": str})
	tweet_ids = rawdata.tweetId.unique()
	tweet_ids = tweet_ids.tolist()

	if DEBUG:
		tweet_ids = tweet_ids[:400]

	return tweet_ids

def main():
	tweet_ids = get_tweet_ids()

	os.makedirs("downloads/tweets/", exist_ok=True)

	for i in tqdm.trange(0, len(tweet_ids), 100):
		response = fetch_tweets(tweet_ids[i:i+100])
		with open(f"downloads/tweets/{i:05d}.json", "w") as f:
			f.write(response.text)

	merge_outputs()

if __name__ == "__main__":
	main()
