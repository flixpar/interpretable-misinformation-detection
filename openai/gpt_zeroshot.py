import glob
import json
import os
import textwrap
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import openai

basepath = Path(__file__).parent

apikey = open(basepath/"openai-key.secret", "r").read()
openai.api_key = apikey

n_random_tweets = 0
model = "gpt-4"
temperature = 0.2


def main():
	tweet_data = select_tweets()

	results = []
	for tweet in tqdm(tweet_data.itertuples(), total=len(tweet_data)):
		try:
			prompt = generate_prompt(tweet)
			response = query_model(prompt)
			community_note = get_community_note(tweet)
			results.append({
				"tweet": tweet._asdict(),
				"prompt": prompt,
				"response": response,
				"community_note": community_note,
			})
		except Exception as e:
			print("Error with tweet:", tweet.tweetId)
			print(e)

	output_folder = get_output_folder()
	save_output_json(results, output_folder)
	save_output_markdown(results, output_folder)

def generate_prompt(tweet):
	tweet_date = tweet.created_at.split("T")[0]

	prompt_text = "Please determine whether the following tweet contains misinformation or not. First, rate the tweet's overall credibility on a scale from 0 to 10. Also rate your confidence in your answer on a scale from 0 to 10. Then, please explain your reasoning. Make a list of all claims in the tweet and whether you believe them to be true, false, or opinion. Cite specific facts and sources that support or refute the factual claims. Explain all context relevant to the tweet and your explanation. Identify the source of the tweet, their biases, and whether they are credible. Finally, summarize your overall finding in a single sentence."
	prompt = f"""\
		{prompt_text}
		Tweet: {tweet.content}
		Date: {tweet_date}
		User: {tweet.user_name}
		User ID: {tweet.user_username}\
	"""
	prompt = textwrap.dedent(prompt)

	return prompt

def query_model(prompt):
	response = openai.ChatCompletion.create(
		model=model,
		temperature=temperature,
		messages=[{"role": "user", "content": prompt}],
	)
	text_response = response.choices[0].message.content
	return text_response

def select_tweets():
	tweet_data = pd.read_csv(basepath/"../data/community-notes/processed/community_notes_tweets.csv", dtype={"tweetId": str})

	if n_random_tweets == 0:
		examples = [
			"1347724256323497989",
			"1374372252179779594",
			"1374210262123704327",
			"1374514885279707144",
			"1426619110742532101",
			"1565139542000246784",
			"1565501546791665668",
			"1564799818836443138",
			"1405549558290169857",
			"1355014457626423300",
			"1405871098705682438",
			"1552694071302422534",
			"1588645415326732290",
			"1578208067719659521",
			"1599961830423728128",
			"1371179956097667081",
			"1371214464498946049",
			"1355721815977582594",
			"1385955098715729926",
			"1566041186581757953",
		]
	else:
		examples = tweet_data.tweetId.sample(n_random_tweets, random_state=0).tolist()

	tweet_data = tweet_data[tweet_data.tweetId.isin(examples)]
	return tweet_data

def get_community_note(tweet):
	data = pd.read_csv(basepath/"../data/community-notes/processed/community_notes.csv", dtype={"tweetId": str})
	community_note = data[data.tweetId == tweet.tweetId].iloc[0]
	community_note = community_note.to_dict()
	return community_note

def get_output_folder():
	prev_folders = glob.glob(f"{basepath}/results/v*/")
	prev_versions = [int(folder.split("/")[1][1:]) for folder in prev_folders]
	prev_version = max(prev_versions) if prev_versions else 0
	output_folder = basepath / f"results/v{prev_version + 1}/"
	os.mkdir(output_folder)
	return output_folder

def save_output_json(results, output_folder):
	with open(f"{output_folder}/results.json", "w") as f:
		json.dump(results, f, indent="\t")

def save_output_markdown(results, output_folder):
	with open(f"{output_folder}/results.md", "w") as f:
		f.write("# Results\n\n")
		for result in results:

			tweet = result["tweet"]
			prompt = result["prompt"]
			response = result["response"]
			community_note = result["community_note"]

			f.write(f"**Tweet:**\n")
			f.write(f"```\n{tweet['content']}\n```\n")
			f.write(f"User: `{tweet['user_name']}`, date: `{tweet['created_at'][:10]}`\n\n")
			f.write(f"**Prompt:**\n")
			f.write(f"```\n{prompt}\n```\n\n")
			f.write(f"**Response:**\n")
			f.write(f"```\n{response}\n```\n\n")
			f.write(f"**Community Note:**\n")
			f.write(f"```\n{community_note['summary']}\n\n")
			for attribute in ["score", "misleading", "believable", "harmful", "factual", "satire", "missing_context", "manipulated_media", "opinion"]:
				f.write(f"{attribute}: {community_note[attribute]}\n")
			f.write("```\n\n")
			f.write("---\n\n")

if __name__ == "__main__":
	main()
