import pandas as pd
import requests
import os
import tqdm
import glob
import json
import time


DEBUG = False

url = "https://api.twitter.com/2/tweets/search/recent"

bearer_token = os.environ["TWITTER_BEARER_TOKEN"]
headers = {"Authorization": f"Bearer {bearer_token}"}

payload_base = {
	"tweet.fields": "created_at,public_metrics,conversation_id,entities,context_annotations",
	"expansions": "author_id,entities.mentions.username,referenced_tweets.id",
	"user.fields": "name,username,verified,profile_image_url",
	"max_results": 100,
}

sites = {
	("nytimes.com", 2000),
	("washingtonpost.com", 1000),
	("cnn.com", 1000),
	("bbc.com", 1000),
	("npr.org", 1000),
	("wikipedia.org", 1000),
}

def fetch_tweets(site, count):
	next_token = None
	while count > 0:
		payload = payload_base.copy()
		payload["query"] = f"url:{site} lang:en -is:retweet -is:reply -is:quote"
		payload["max_results"] = min(count, 100)
		payload["next_token"] = next_token
		response = requests.request("GET", url, headers=headers, data=payload)

		count -= payload["max_results"]
		next_token = response.json().get("meta", {}).get("next_token")

		ms = round(time.time() * 1000)
		with open(f"downloads/tweets/{ms}.json", "w") as f:
			f.write(response.text)

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

def main():
	os.makedirs("downloads/tweets/", exist_ok=True)

	for site, count in tqdm.tqdm(sites):
		fetch_tweets(site, count)

	merge_outputs()

if __name__ == "__main__":
	main()
