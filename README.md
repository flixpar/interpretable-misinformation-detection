# Interpretable Misinformation Detection
Felix Parker, Kristen Nixon, Sonia Jindal

This project develops an interpretable system for detecting misinformation on Twitter. We train models that use the content of a tweet and its metadata to classify it as either misleading or not misleading, along with a corresponding confidence score, and provide various interpretations of the predictions. We construct a new dataset for this purpose from subset of the Twitter Community Notes dataset and additional news-related tweets.

## Usage
To run our system first install the required packages in requirements.txt. Then run the scripts in this repository in the following order:

**Data Processing:**
1. data/community-notes/community_notes.jl
2. data/community-notes/fetch_tweets.py
3. data/community-notes/format_tweets.py
4. data/news-tweets/fetch_news_tweets.py
5. data/news-tweets/format_tweets.py
6. data/combined/combine_datasets.jl
7. data/combined/generate_splits.py
8. data/twitter-users/get_users.py
9. data/twitter-users/format_users.py

**Models:**
1. models/engagementscore/engagement-model.py
2. models/userscore/user-model.py
3. models/linkscore/fetch-linkscores.py
4. models/linkscore/link-model.py
5. models/textscore/textscore_train.py
6. models/textscore/textscore_inference.py
7. ?

**User Study:**
1. userstudy/data/fetch_tweets.py
2. userstudy/data/format_tweets.py
3. userstudy/backend.py
4. userstudy/analysis/database_to_csv.py
