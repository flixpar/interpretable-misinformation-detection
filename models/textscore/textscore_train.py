import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import GPT2Config
from datasets import Dataset

import evaluate

import time
import os
from pathlib import Path


# parameters

basepath = Path("../../")
SAVEID = round(time.time() * 1000)

EPOCHS = 10
BATCH_SIZE = 16
GRAD_STEPS = 2
MAX_LENGTH = 64

FP16 = False
GRAD_CHECKPOINT = False

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

train_df = data_df.loc[data_df.tweetId.isin(train_ids)]
val_df = data_df.loc[data_df.tweetId.isin(val_ids)]

train_data = data_df.to_dict("records")
val_data = val_df.to_dict("records")

dataset_train = Dataset.from_list(train_data)
dataset_val = Dataset.from_list(val_data)

dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_val = dataset_val.map(tokenize_function, batched=True)

# model setup
print("Setting up model...")

model_config = GPT2Config(
    block_size=MAX_LENGTH,
    num_labels=2,
)

model = AutoModelForSequenceClassification.from_pretrained("gpt2", config=model_config)
# model = AutoModelForSequenceClassification.from_pretrained("gpt2", problem_type="multi_label_classification")

model.config.pad_token_id = model.config.eos_token_id

# setup trainer
print("Setting up trainer...")

eval_metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_eval_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return eval_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_STEPS,
    gradient_checkpointing=GRAD_CHECKPOINT,
    fp16=FP16,

    evaluation_strategy="epoch",

    save_strategy="epoch",
    load_best_model_at_end=True,
    output_dir=f"model_saves/{SAVEID}/",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_eval_metrics,
)

# train
print("Training...")

training_result = trainer.train()


if SHUTDOWN:
    os.system("sudo shutdown now")
