"""
# _* coding: utf8 *_

filename: dataset.py

@author: sounishnath
createdAt: 2022-12-13 21:37:27
"""

import typing

import torch
from tokenizer import Tokenizer


class IntentDataset:
    def __init__(self, intents, tags, tokenizer: Tokenizer) -> None:
        self.intents = intents
        self.tags = tags
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx: int):
        tokenized_dict = self.tokenizer.tokenize(self.intents[idx])
        return {
            "token_type_ids": torch.tensor(
                tokenized_dict["token_type_ids"], dtype=torch.long
            ),
            "tag": torch.tensor(self.tags[idx], dtype=torch.long),
        }
