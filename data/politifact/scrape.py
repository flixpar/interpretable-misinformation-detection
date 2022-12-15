import requests
from bs4 import BeautifulSoup
from cleantext import clean
import pandas as pd
import numpy as np
import tqdm
import time

N = 5000


def get_urls_page(k):
	list_url = f"https://www.politifact.com/factchecks/list/?page={k}"
	list_page = requests.get(list_url)
	list_soup = BeautifulSoup(list_page.content, "html.parser")

	base_url = "https://www.politifact.com"
	urls = [base_url + t.find("a")["href"] for t in list_soup.find_all(class_="m-statement__quote")]

	return urls

def get_urls(n):
	urls = []

	page = 1
	while n > 0:
		page_urls = get_urls_page(page)
		urls.extend(page_urls)
		n -= len(page_urls)
		page += 1

	return urls

urls = get_urls(N)

urls_df = pd.DataFrame(urls, columns=["url"])
urls_df.to_csv(f"politifact-urls-{N}.csv", index=False)

def clean_text(text):
	return clean(
		text,
		fix_unicode=True,
		to_ascii=True,
		lower=False,
		no_line_breaks=True,
		no_urls=True, replace_with_url="",
		no_punct=False,
		lang="en",
	).replace("* ", "")

def parse_page(url):
	page = requests.get(url)
	soup = BeautifulSoup(page.content, "html.parser")

	statement = clean_text(soup.find(class_="m-statement__quote").text)
	person = soup.find(class_="m-statement__name").text.strip()
	label = soup.find(class_="m-statement__meter").find(class_="c-image__thumb")["alt"]

	try:
		summary = clean_text(soup.find(class_="short-on-time").text)
	except:
		summary = ""

	try:
		ruling_parts = soup.find(text="Our ruling").parent.parent.find_next_siblings()
		if len(ruling_parts) == 0: raise Exception
	except:
		try:
			ruling_parts = list(soup.find(text="Our ruling").parent.next_siblings)
			if len(ruling_parts) == 0: raise Exception
		except:
			try:
				ruling_parts = soup.find("article", class_="m-textblock").find_all("p")[-4:]
			except:
				ruling_parts = []
	ruling = clean_text(" ".join([p.text for p in ruling_parts]))

	return {
		"statement": statement,
		"person": person,
		"label": label,
		"summary": summary,
		"ruling": ruling,
	}

data = []
for url in tqdm.tqdm(urls):
	try:
		d = parse_page(url)
		data.append(d)
	except:
		print(f"Error on {url}")

data_df = pd.DataFrame(data)
data_df.to_csv(f"politifact-{N}.csv", index=False)
