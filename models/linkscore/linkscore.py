import pandas as pd
import json

import requests
import tldextract
import re

import multiprocessing
import tqdm

import os
from pathlib import Path
basepath = Path(__file__).parent

DEBUG = False
PROCESSES = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"}


def main():
	print("Loading data...")
	with open(basepath.joinpath("../../data/combined/processed/tweets_cn_news.json"), "r") as f:
		tweets = json.load(f)["data"]

	if DEBUG:
		tweets = tweets[:100]

	all_urls = [get_urls(tweet) for tweet in tweets]
	all_urls = [url for urls in all_urls for url in urls]
	all_urls = list(set(all_urls))
	all_urls = [url for url in all_urls if not is_twitter_url(url)]

	print("Expanding URLs...")
	expanded_urls = pmap(expand_url, all_urls)

	expansions_df = pd.DataFrame(expanded_urls)
	expansions_df.to_csv(basepath.joinpath("downloads/url_expansions.csv"), index=False)

	domains = [url["domain"] for url in expanded_urls]
	domains = list(set(domains))

	print("Fetching domain scores...")
	domain_scores = pmap(fetch_domain_score, domains)
	domain_scores_lookup = {domain_score["domain"]: domain_score for domain_score in domain_scores}

	print("Computing link scores...")
	link_scores_lookup = {url["url"]: domain_scores_lookup[url["domain"]] for url in expanded_urls}

	default_link_score = {"score": 0, "confidence": 0}
	def avg_link_score(tweet):
		if "entities" in tweet and "urls" in tweet["entities"] and tweet["entities"]["urls"]:
			urls = [u["expanded_url"] for u in tweet["entities"]["urls"]]
			scores = [link_scores_lookup.get(url, default_link_score) for url in urls]
			avg_score = {
				"tweetId": tweet["id"],
				"score": mean([s["score"] for s in scores]),
				"confidence": mean([s["confidence"] for s in scores]),
			}
			return avg_score
		else:
			avg_score = {"tweetId": tweet["id"], **default_link_score}
			return avg_score

	link_scores = [avg_link_score(tweet) for tweet in tweets]
	link_scores_df = pd.DataFrame(link_scores)

	print("Saving results...")
	os.makedirs(basepath.joinpath("downloads"), exist_ok=True)
	link_scores_df.to_csv(basepath.joinpath("downloads/link_scores.csv"), index=False)

	print("Done!")

def pmap(func, iterable):
	with multiprocessing.Pool(PROCESSES) as pool:
		return list(tqdm.tqdm(pool.imap(func, iterable), total=len(iterable)))

def mean(xs):
	return sum(xs) / len(xs)

def is_twitter_url(url):
	if "https://twitter.com" in url:
		return True
	elif "https://t.co" in url:
		return True
	else:
		return False

def expand_url(url):
	expanded_url = expand_url_(url)
	domain = tldextract.extract(expanded_url).registered_domain
	return {"url": url, "expanded_url": expanded_url, "domain": domain}

def expand_url_(url):
	try:
		r = requests.head(url, allow_redirects=True, timeout=10, headers=HEADERS)
		return r.url
	except Exception as e:
		try:
			u = str(e)
			u = re.search(r"host='(.*)'", u).group(1)
			return f"https://{u}"
		except:
			return url

def get_urls(tweet):
	if "entities" in tweet and "urls" in tweet["entities"]:
		urls = tweet["entities"]["urls"]
		if urls:
			return [url["expanded_url"] for url in urls]
	return []

def fetch_domain_score(domain):
	r = requests.get("https://misinfo.me/misinfo/api/credibility/sources/", params={"source": domain}).json()
	return {
		"domain": domain,
		"score": (r["credibility"]["value"] * 2) - 1,
		"confidence": r["credibility"]["confidence"],
	}

if __name__ == "__main__":
	main()
