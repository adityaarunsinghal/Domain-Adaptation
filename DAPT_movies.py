#!/usr/bin/env python

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import pandas as pd
import os.path
from os import path

print("-------------ALL IMPORTED------------")

paths = [str(x) for x in Path("./data/simple_movie_text_MLM/").glob("**/*.txt")]

if( not (path.exists("data/simple_movie_text_MLM/ALL_DATA.txt"))):
    alldata = open("data/simple_movie_text_MLM/ALL_DATA.txt", "w")
    for each_path in paths:
        f = open(each_path, "r")
        alldata.write(f.read() + "\n")
    print("-------------Big Data Text File Made------------")

else:
    print("-------------Big Data Text File WAS ALREADY MADE------------")

Path("movie_roberta/roberta_DAPT_movies_model").mkdir(parents=True, exist_ok=True)

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("movie_roberta/roberta_DAPT_movies_model")

print("-------------TOKENIZER SAVED------------")

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./roberta_DAPT_movies_model/vocab.json",
    "./roberta_DAPT_movies_model/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

import torch
print("CUDA is available: " , torch.cuda.is_available())

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./roberta_DAPT_movies_model", max_len=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

print("Num of parameters = ", model.num_parameters())

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/simple_movie_text_MLM/ALL_DATA.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./roberta_DAPT_movies_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print("-------------STARTING TO TRAIN------------")

trainer.train()

print("-------------TRAINING ENDED------------")

trainer.save_model("./roberta_DAPT_movies_model")

print("-------------MODEL SAVED------------")

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./roberta_DAPT_movies_model",
    tokenizer="./roberta_DAPT_movies_model"
)

print("-------------COMPLETED------------")
