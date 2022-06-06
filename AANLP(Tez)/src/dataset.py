"""
# _* coding: utf8 *_

filename: dataset.py

@author: sounishnath
createdAt: 2022-06-06 18:21:25
"""

import torch

import config


class Dataset:
    def __init__(self, text, sentiment) -> None:
        self.text = text
        self.sentiment = sentiment
        self.tokenizer = config.TOKENIZER
        self.max_length = config.MAX_LENGTH

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        tokenized = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        ids = tokenized.get("input_ids")
        attention_mask = tokenized.get("attention_mask")
        token_type_ids = tokenized.get("token_type_ids")

        return dict(
            ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
            target=torch.tensor(self.sentiment[index], dtype=torch.long),
        )
