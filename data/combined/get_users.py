import pandas as pd
import requests
import os
import tqdm
import glob
import json


DEBUG = False

url = "https://api.twitter.com/2/users"

bearer_token = os.environ["TWITTER_BEARER_TOKEN"]
headers = {"Authorization": f"Bearer {bearer_token}"}

payload_base = {
	"user.fields": "id,created_at,public_metrics,verified",
}

def fetch_users(user_ids):
	payload = payload_base.copy()
	payload["ids"] = ",".join(user_ids)
	response = requests.request("GET", url, headers=headers, data=payload)
	return response

def merge_outputs():
	fns = glob.glob("downloads/users/*.json")

	data = {"data": []}
	errors = []

	for fn in fns:
		with open(fn, "r") as f:
			f_data = json.load(f)

			if "data" in f_data:
				data["data"].extend(f_data["data"])
			else:
				print("No data in:", fn)

			if "errors" in f_data:
				errors.extend([e["resource_id"] for e in f_data["errors"]])

	with open("downloads/users.json", "w") as f:
		json.dump(data, f, indent="\t")

	with open("downloads/user-errors.json", "w") as f:
		json.dump(errors, f, indent="\t")

def get_user_ids():
	rawdata = pd.read_csv("processed/tweets_cn_news.csv", dtype={"userId": str})
	user_ids = rawdata.userId.unique()
	user_ids = user_ids.tolist()

	if DEBUG:
		user_ids = user_ids[:400]

	return user_ids

def main():
	user_ids = get_user_ids()

	os.makedirs("downloads/users/", exist_ok=True)

	for i in tqdm.trange(0, len(user_ids), 100):
		response = fetch_users(user_ids[i:i+100])
		with open(f"downloads/users/{i:05d}.json", "w") as f:
			f.write(response.text)

	merge_outputs()

if __name__ == "__main__":
	main()
