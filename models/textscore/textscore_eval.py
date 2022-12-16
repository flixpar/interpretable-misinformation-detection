import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

import evaluate

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
data_df.rename(columns={"content": "text", "misleading": "label"}, inplace=True)

val_df = data_df.loc[data_df.tweetId.isin(val_ids)]
val_data = val_df.to_dict("records")

dataset_val = Dataset.from_list(val_data)
dataset_val = dataset_val.map(tokenize_function, batched=True)

# model setup
print("Setting up model...")

model = AutoModelForSequenceClassification.from_pretrained(f"model_saves/{SAVEID}/{SAVE_CHECKPOINT}/")
# model = AutoModelForSequenceClassification.from_pretrained("gpt2", problem_type="multi_label_classification")

# setup evaluator
print("Setting up evaluator...")

eval_metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_eval_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return eval_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    per_device_eval_batch_size=BATCH_SIZE,
    fp16=FP16,
    output_dir="./",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_val,
    eval_dataset=dataset_val,
    compute_metrics=compute_eval_metrics,
)

# eval
print("Evaluating...")

metrics = trainer.evaluate(eval_dataset=dataset_val)
print(metrics)


if SHUTDOWN:
    os.system("sudo shutdown now")
