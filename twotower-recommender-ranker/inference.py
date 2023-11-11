import argparse
import logging
import os
import random
from dataclasses import dataclass

import pandas as pd
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from twotower_recommender_ranker import (
    Logger,
    MovieLenseDataset,
    TwoTowerRecommenderRankerLitModel,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = TwoTowerRecommenderRankerLitModel.load_from_checkpoint(
        "/Users/sounishnath/workspaces/torchf/twotower-recommender-ranker/lightning_logs/twotower_model/version_0/checkpoints/beginner-swan-3465.ckpt",
        lr=3e-4,
        dropout=0.30,
    )
    trainer = pl.Trainer()

    dataset = MovieLenseDataset(
        dataset_directory="./datasets/ml-100k",
        tokenizer_name="distilbert-base-uncased",
        n_rows=1500,
    )
    Logger.get_logger().debug(
        "total unique rating = {}".format(
            dataset.data_user_item_dfx.loc[:, "rating"].unique()
        )
    )
    train_dataset, val_dataset = dataset.train_test_split(
        dataset=dataset, val_size=0.25
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    val_dataloader = DataLoader([val_dataset[-1]], batch_size=2, shuffle=False)

    outs = trainer.predict(model, val_dataloader)

    Logger.get_logger().info(next(iter(val_dataloader)))
    Logger().get_logger().info(outs[0]["rating"])
