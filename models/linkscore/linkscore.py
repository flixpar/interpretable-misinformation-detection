import pandas as pd
import json

import requests
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects
from urllib.parse import urlsplit
import re

import multiprocessing
import tqdm

PROCESSES = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"}


def main():
	print("Loading data...")
	data = pd.read_csv("../../data/combined/processed/combined.csv")
	tweet_ids = data.tweetId.unique().tolist()

	with open("../../data/combined/processed/tweets_cn_news.json", "r") as f:
		tweets = json.load(f)

	all_urls = [get_urls(tweet) for tweet in tweets]
	all_urls = [url for urls in all_urls for url in urls]

	print("Expanding URLs...")
	expanded_urls = pmap(expand_url, all_urls)

	print("Fetching link scores...")
	link_scores_list = pmap(fetch_link_score, expanded_urls)
	link_scores_lookup = {link_score["url"]: link_score for link_score in link_scores_list}

	def avg_link_score(tweet):
		if "entities" in tweet and "urls" in tweet["entities"]:
			urls = tweet["entities"]["urls"]
			if urls:
				url = urls[0]["expanded_url"]
				return fetch_link_score(url)
		urls = tweet["entities"]["urls"]

	link_scores = [{"tweetId": tweet["id"], "linkscore": link_score(tweet)} for tweet in tweets]
	link_scores_df = pd.DataFrame(link_scores)
	link_scores_df.to_csv("downloads/link_scores.csv", index=False)

def pmap(func, iterable):
	with multiprocessing.Pool(PROCESSES) as pool:
		return list(tqdm.tqdm(pool.imap(func, iterable), total=len(iterable)))

def expand_url(url):
	try:
		r = requests.head(url, allow_redirects=True, timeout=10, headers=HEADERS)
		return r.url
	except (ConnectionError, ReadTimeout, TooManyRedirects) as e:
		try:
			u = str(e)
			u = re.search(r"host='(.*)'", u).group(1)
			return f"https://{u}"
		except:
			return url
	except:
		return url

def get_urls(tweet):
	if "entities" in tweet and "urls" in tweet["entities"]:
		urls = tweet["entities"]["urls"]
		if urls:
			return [url["expanded_url"] for url in urls]
	return []

def fetch_link_score(url):
	domain = urlsplit(url).netloc
	r = requests.get("https://misinfo.me/misinfo/api/credibility/sources/", params={"source": domain}).json()
	return {"url": url, "domain": domain, "score": r["credibility"]["value"], "confidence": r["credibility"]["confidence"]}

if __name__ == "__main__":
	main()
