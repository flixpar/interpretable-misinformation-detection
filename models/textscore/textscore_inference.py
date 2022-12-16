import numpy as np
import pandas as pd
import scipy.special

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

import os
from pathlib import Path


# parameters

SAVEID = "1671089396783"
SAVE_CHECKPOINT = "checkpoint-3700"

basepath = Path("../../")

BATCH_SIZE = 8
MAX_LENGTH = 64
FP16 = False

SHUTDOWN = False

# data processing
print("Loading data...")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

splits_df = pd.read_csv(basepath.joinpath("data/combined/processed/splits.csv"))
train_ids = set(splits_df.loc[splits_df.split == "train"].tweetId.tolist())
val_ids = set(splits_df.loc[splits_df.split == "val"].tweetId.tolist())

data_df = pd.read_csv(basepath.joinpath("data/combined/processed/tweets_cn_news.csv"))
data_df = data_df[["tweetId", "content", "misleading"]]
data_df["split"] = data_df.tweetId.apply(lambda x: "train" if x in train_ids else "val")
data_df.rename(columns={"content": "text", "misleading": "label"}, inplace=True)
data = data_df.to_dict("records")

dataset_all = Dataset.from_list(data)
dataset_all = dataset_all.map(tokenize_function, batched=True)

# model setup
print("Setting up model...")

model = AutoModelForSequenceClassification.from_pretrained(f"model_saves/{SAVEID}/{SAVE_CHECKPOINT}/")
# model = AutoModelForSequenceClassification.from_pretrained("gpt2", problem_type="multi_label_classification")

# setup inference
print("Setting up inference...")

training_args = TrainingArguments(
    per_device_eval_batch_size=BATCH_SIZE,
    fp16=FP16,
    output_dir="./",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_all,
    eval_dataset=dataset_all,
)

# eval
print("Inference...")

results = trainer.predict(test_dataset=dataset_all)

ids = [ex["tweetId"] for ex in dataset_all]
splits = [ex["split"] for ex in dataset_all]

labels = results.label_ids
preds = results.predictions.argmax(axis=1)
pred_scores = scipy.special.softmax(results.predictions, axis=1)
pred_scores = pred_scores[np.arange(len(preds)),preds]

labels = (labels * 2) - 1
preds = (preds * 2) - 1
pred_scores = pred_scores * preds

output_list = [{"tweetId": r[0], "split": r[1], "label": r[2], "pred": r[3], "pred_score": r[4]} for r in zip(ids, splits, labels, preds, pred_scores)]

output_df = pd.DataFrame(output_list)
output_df.to_csv(f"textscores_{SAVEID}.csv", index=False)


if SHUTDOWN:
    os.system("sudo shutdown now")
