"""
# _* coding: utf8 *_

filename: config.py

@author: sounishnath
createdAt: 2022-06-06 18:15:51
"""

import transformers

DEVICE='cpu'
EPOCHS=10
TRAIN_BATCH_SIZE=32
VALID_BATCH_SIZE=32
MAX_LENGTH=128
ACCUMULATION_STEP=1
LEARNING_RATE=5e-5
BERT_PATH='bert-base-uncased'
TRAINING_DATASET='../input/Tweets.csv'
MODEL_PATH='model.bin'
TOKENIZER=transformers.AutoTokenizer.from_pretrained(BERT_PATH)
