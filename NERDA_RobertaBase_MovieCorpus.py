#!/usr/bin/env forCCM/bin/python

import joblib
import pandas as pd
import NERDA
import transformers

training = joblib.load("data/MIT_movie/dict_structure/training.dict")
validation = joblib.load("data/MIT_movie/dict_structure/validation.dict")

tag_scheme = ['B-ACTOR',
 'B-CHARACTER',
 'B-DIRECTOR',
 'B-GENRE',
 'B-PLOT',
 'B-RATING',
 'B-RATINGS_AVERAGE',
 'B-REVIEW',
 'B-SONG',
 'B-TITLE',
 'B-TRAILER',
 'B-YEAR',
 'I-ACTOR',
 'I-CHARACTER',
 'I-DIRECTOR',
 'I-GENRE',
 'I-PLOT',
 'I-RATING',
 'I-RATINGS_AVERAGE',
 'I-REVIEW',
 'I-SONG',
 'I-TITLE',
 'I-TRAILER',
 'I-YEAR',
 'O']

# hyperparameters for network

dropout = 0.1
transformer = 'distilroberta-base'

# hyperparameters for training
training_hyperparameters = {'epochs' : 4,
                            'warmup_steps' : 500,
                            'train_batch_size': 13,
                            'learning_rate': 0.0001
                            }

from NERDA.models import NERDA

model = NERDA(
dataset_training = training,
dataset_validation = validation,
tag_scheme = tag_scheme, 
tag_outside = 'O',
transformer = transformer,
dropout = dropout,
hyperparameters = training_hyperparameters
)

model.train()
