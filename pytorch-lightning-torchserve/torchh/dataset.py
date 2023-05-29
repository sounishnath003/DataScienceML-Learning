"""
# _* coding: utf8 *_

filename: dataset.py

@author: sounishnath
createdAt: 2023-05-23 00:13:45
"""

import pytorch_lightning as pl
import transformers
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import AutoTokenizer

import torch
import torch.utils.data

__MAX_LENGTH__ = 64


class ImdbDataset(object):
    def __init__(
        self,
        review,
        sentiment,
        max_length: int = __MAX_LENGTH__,
        tokenizer: AutoTokenizer = "distilbert-base-uncased",
    ) -> None:
        super(ImdbDataset, self).__init__()
        self._review = review
        self._sentiment = sentiment
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return len(self._sentiment)

    def __getitem__(self, id: int):
        review = self._review[id]
        sentiment = 1 if self._review[id] == "positive" else 0

        _tokenized = self._tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        _tokenized = {
            "input_ids": _tokenized.get("input_ids").squeeze(0),
            "token_type_ids": _tokenized.get("token_type_ids").squeeze(0),
            "attention_mask": _tokenized.get("attention_mask").squeeze(0),
        }

        return {
            **_tokenized,
            "sentiment": torch.tensor(sentiment, dtype=torch.float),
        }


class LitImdbDataloader(pl.LightningDataModule):
    def __init__(
        self,
        train_dfx,
        valid_dfx,
        train_bs: int = 32,
        valid_bs: int = 8,
    ) -> None:
        super(LitImdbDataloader, self).__init__()
        self._train_dfx = train_dfx
        self._valid_dfx = valid_dfx
        self._train_bs = train_bs
        self._valid_bs = valid_bs

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = ImdbDataset(
            review=self._train_dfx["review"].values,
            sentiment=self._train_dfx["sentiment"].values,
            max_length=__MAX_LENGTH__,
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self._train_bs,
            num_workers=1,
            collate_fn=transformers.DefaultDataCollator(),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = ImdbDataset(
            review=self._valid_dfx["review"].values,
            sentiment=self._valid_dfx["sentiment"].values,
            max_length=__MAX_LENGTH__,
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=self._valid_bs,
            num_workers=1,
            collate_fn=transformers.DefaultDataCollator(),
        )
