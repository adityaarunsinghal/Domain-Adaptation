#!/usr/bin/env python

from pathlib import Path
import os.path
from os import path

print("-------------ALL IMPORTED------------")

paths = [str(x) for x in Path("./data/simple_movie_text_MLM/").glob("**/*.txt")]
train_test_split = round(len(paths)*0.8)
train_paths = paths[:train_test_split]
test_paths = paths[train_test_split:]

if( not (path.exists("data/simple_movie_text_MLM/ALL_DATA_train.txt"))):
    alldata = open("data/simple_movie_text_MLM/ALL_DATA_train.txt", "w")
    for each_path in train_paths:
        f = open(each_path, "r")
        alldata.write(f.read() + "\n")
    print("-------------Big Data Train Text File Made------------")

else:
    print("-------------Big Data Train Text File WAS ALREADY MADE------------")

if( not (path.exists("data/simple_movie_text_MLM/ALL_DATA_test.txt"))):
    alldata = open("data/simple_movie_text_MLM/ALL_DATA_test.txt", "w")
    for each_path in test_paths:
        f = open(each_path, "r")
        alldata.write(f.read() + "\n")
    print("-------------Big Data Test Text File Made------------")

else:
    print("-------------Big Data Test Text File WAS ALREADY MADE------------")

Path("movie_roberta/roberta_DAPT_movies_model").mkdir(parents=True, exist_ok=True)