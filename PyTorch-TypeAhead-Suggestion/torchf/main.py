"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-07-26 09:55:32
"""

import logging
import re
import json
import warnings
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import AUROC, Accuracy, F1Score
from tqdm import tqdm
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")


def log(data):
    logging.basicConfig(level=logging.INFO)
    logging.info(10 * "=========")
    logging.info({"payload": data})
    logging.info(10 * "=========")


class Tokenizer:
    def __init__(
        self, tokenizer_model_name="bert-base-uncased", max_length: int = 32
    ) -> None:
        self._tokenizer_name = tokenizer_model_name
        self._max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

        self.vocabs = set(["<PAD>", "<UNK>"])
        self._word_2_id = {"<PAD>": 0, "<UNK>": 1}
        self._id_2_word = {0: "<PAD>", 1: "<UNK>"}
        self._pad_token_id = 0

    @property
    def all_vocabs(self):
        # return list(self.tokenizer.vocab.keys())
        return list(self._word_2_id.keys())

    @property
    def pad_token_id(self):
        # return self.tokenizer.pad_token_id
        return self._pad_token_id

    def encode(self, text: str):
        # return self.tokenizer.encode(
        #     text,
        #     add_special_tokens=True,
        # )
        text = re.sub("[!@#$%^&*()']", "", text)
        tokens = []
        for word in re.split("\W+", text.lower()):
            if not word in self._word_2_id:
                self._word_2_id[word] = len(self._word_2_id) + 1
                self._id_2_word = {v: k for k, v in self._word_2_id.items()}
            tokens.append(self._word_2_id.get(word, self.pad_token_id))
        return tokens

    def decode(self, tokens):
        # return self.tokenizer.decode(
        #     token_ids=tokens,
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=True,
        # )
        words = []

        for tok in tokens:
            if tok != self.pad_token_id:
                words.append(self._id_2_word.get(tok, "<CLS>"))
        return words

    def save(self):
        with open('word_2_id.json', 'w') as f: json.dump(self._word_2_id, f)
        with open('id_2_word.json', 'w') as f: json.dump(self._id_2_word, f)

class Seq2SeqDataset:
    def __init__(self, data, targets) -> None:
        self._data = data
        self._targets = targets

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return dict(
            tokens=torch.tensor(
                self._data[item],
                dtype=torch.long,
            ),
            target=torch.tensor(
                self._targets[item],
                dtype=torch.long,
            ),
        )


class BaseSequence2SequenceModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 150,
        pad_token_id: int = 0,
        dropout: float = 0.23,
        learning_rate: float = 3e-4,
    ) -> None:
        super(BaseSequence2SequenceModel, self).__init__()
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._pad_token_id = pad_token_id
        self._learning_rate = learning_rate

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=pad_token_id,
            scale_grad_by_freq=True,
        )
        torch.nn.init.xavier_uniform_(self.embedding.weight)

        self.dropout = nn.Dropout(dropout)
        self._lstm_embedding_size = int(embedding_size // 6)
        self.gated_rnn = nn.LSTM(
            embedding_size,
            hidden_size=self._lstm_embedding_size,
            num_layers=2,
            batch_first=True,
        )
        self.sequence_predictor = nn.Linear(self._lstm_embedding_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.save_hyperparameters()

    def forward(self, tensors):
        # log(tensors.size())
        logits = self.embedding(tensors)
        # log(logits.size())
        logits = self.dropout(F.relu(logits))
        # log(logits.size())

        hidd_ = torch.zeros(2, tensors.size(0), self._lstm_embedding_size).to(
            self._device
        )
        cell_ = torch.zeros(2, tensors.size(0), self._lstm_embedding_size).to(
            self._device
        )
        logits, hidd_ = self.gated_rnn(logits, (hidd_, cell_))
        # log(hidd_.size())

        logits = self.sequence_predictor(logits)
        logits = logits[:, -1, :]
        # log(logits.size())
        logits = self.softmax(logits)
        # log(logits.size())
        return logits

    def _common_network_steps(self, data, targets):
        logits = self.forward(data)
        loss = nn.NLLLoss()(logits, targets)
        acc = Accuracy("multiclass", num_classes=self._vocab_size)(
            logits.to("cpu"), targets.to("cpu")
        )

        return (
            logits,
            loss,
            [
                {"metric_type": "acc", "score": acc},
            ],
        )

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        data = batch["tokens"]
        targets = batch["target"]

        logits, loss, metrics = self._common_network_steps(data, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for metric in metrics:
            self.log(
                metric["metric_type"],
                metric["score"],
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )

        return loss

    def validation_step(
        self, batch, batch_idx, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        data = batch["tokens"]
        targets = batch["target"]

        logits, loss, metrics = self._common_network_steps(data, targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for metric in metrics:
            self.log(
                metric["metric_type"],
                metric["score"],
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )

        return loss

    def predict_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        data = batch["tokens"]
        targets = batch["target"]

        logits, loss, metrics = self._common_network_steps(data, targets)
        return logits

    def configure_optimizers(self) -> Any:
        opt = torch.optim.SGD(
            self.parameters(), lr=self._learning_rate, momentum=1e-4, weight_decay=1e-3
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.1, min_lr=5e-8, verbose=True
        )
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val_loss"}


def create_sequence_tokens(
    tokenizer: Tokenizer, lines: list[str], pad_length: int = 64, ngrams: int = 3
):
    data = []
    targets = []

    for line in tqdm(lines, total=len(lines)):
        tokens = tokenizer.encode(line)
        for i in range(1, len(tokens)):
            target_ngrams = tokens[: i + ngrams][:pad_length]
            target_length = len(target_ngrams)
            if target_length < pad_length:
                pads = [tokenizer.pad_token_id] * (pad_length - target_length)
                target_ngrams = pads + target_ngrams
                target_ngrams = target_ngrams[:pad_length]
                if len(target_ngrams) != pad_length:
                    log(
                        "pad length = {0} total length={1}".format(
                            len(pads), len(target_ngrams)
                        )
                    )
            data.append(target_ngrams[:-1])
            targets.append(target_ngrams[-1])

    return data, targets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(111)

    rawdata = open("data/data.txt").readlines()
    log("total paragraphs {0}".format(len(rawdata)))

    tokenizer = Tokenizer()

    data, targets = create_sequence_tokens(tokenizer, rawdata, pad_length=64, ngrams=3)
    log("data size={0}, target_size={1}".format(len(data), len(targets)))
    tokenizer.save()

    dataset = Seq2SeqDataset(data=data, targets=targets)
    train_ratio = 0.80
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, lengths=[train_ratio, 1 - train_ratio]
    )

    log("Training dataset={0}".format(len(train_dataset)))
    log("Validation dataset={0}".format(len(valid_dataset)))

    print(valid_dataset[88])

    seq2seq_lit_model = BaseSequence2SequenceModel(
        vocab_size=len(tokenizer.all_vocabs),
        pad_token_id=tokenizer.pad_token_id,
        learning_rate=1e-3,
    )
    print(seq2seq_lit_model)

    """
    outs = seq2seq_lit_model.forward(dataset[0]["tokens"].unsqueeze(0))
    print(outs)
    print(outs.size())
    exit(-1)
    """

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=55,
        log_every_n_steps=True,
        enable_checkpointing=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
    )
    trainer.fit(
        model=seq2seq_lit_model,
        train_dataloaders=torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=5
        ),
        val_dataloaders=torch.utils.data.DataLoader(
            valid_dataset, shuffle=False, batch_size=64, num_workers=3
        ),
    )
