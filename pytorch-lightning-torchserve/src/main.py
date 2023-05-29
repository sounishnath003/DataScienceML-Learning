"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-05-22 23:37:47
"""

import logging
import os
from dataclasses import dataclass

import pandas as pd
import pytorch_lightning as pl

import torchh
from src import utils


@dataclass
class Config:
    MAX_LENGTH = torchh.dataset.__MAX_LENGTH__
    TRAIN_BS = 64
    VALID_BS = 8
    DATASET_PATH = os.path.join(os.getcwd(), "datasets", "IMDB Dataset.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(seed=31)

    dfx = pd.read_csv(Config.DATASET_PATH)
    dfx.loc[:, "tokens"] = Config.MAX_LENGTH
    dfx["tokens"] = dfx["review"].astype(str).map(len)
    dfx = (
        dfx[dfx["tokens"] >= Config.MAX_LENGTH]
        .copy()
        .drop(["tokens"], axis=1)
        .reset_index(drop=True)
    )
    logging.info(utils.to_string(f"total rows={dfx.shape}"))

    train_dfx, valid_dfx = utils.split_dataset(dfx)
    logging.info(
        utils.to_string(f"train_size={train_dfx.shape}, valid_size={valid_dfx.shape}")
    )
    logging.info(utils.to_string(train_dfx.columns))

    lit_imdb_model = torchh.model.LitImdbNeuralNet(
        foundation_model=torchh.model.ImdbNeuralNet(),
        n_classes=1,
    )
    logging.info(lit_imdb_model)

    trainer = pl.Trainer(
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        max_epochs=2,
        log_every_n_steps=1,
    )
    trainer.fit(
        model=lit_imdb_model,
        datamodule=torchh.dataset.LitImdbDataloader(
            train_dfx=train_dfx, valid_dfx=valid_dfx
        ),
    )
