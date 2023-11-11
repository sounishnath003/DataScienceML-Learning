"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-10-29 01:25:32
"""

import argparse
import logging
import os
import random
from dataclasses import dataclass

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:TwoTowerRankerRecommenderModel:%(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        handlers=[
            logging.FileHandler(filename=os.path.join(os.getcwd(), "activity.log")),
            logging.StreamHandler(),
        ],
    )
    parser = argparse.ArgumentParser(prog="two-tower-recommender-ranker-system")
    parser.add_argument(
        "--datasetFolder",
        type=str,
        required=True,
        help="provide dataset such as movie-lense-100k (ml-100k)",
    )
    parser.add_argument(
        "--modelName", type=str, required=True, help="provide a good model name"
    )
    parser.add_argument(
        "--pretrainedHfModelName",
        type=str,
        required=False,
        default="distilbert-base-uncased",
        help="provide pretrained HF-model-name",
    )
    parser.add_argument(
        "--train_bs",
        type=int,
        required=True,
        help="provide training batch size like 32",
    )
    parser.add_argument(
        "--val_bs",
        type=int,
        required=True,
        help="provide training batch size like 32",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="provide learning rate",
    )
    parser.add_argument("--version", type=str, default=str(random.random())[2:6])
    parser.add_argument("--n_rows", type=int, default=None)

    opts, runtime_args = parser.parse_known_args()
    Logger.get_logger().info(dict(opts=opts, runtime_args=runtime_args))

    dataset = MovieLenseDataset(
        dataset_directory=opts.datasetFolder,
        tokenizer_name=opts.pretrainedHfModelName,
        n_rows=opts.n_rows,
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
        batch_size=opts.train_bs,
        shuffle=True,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=opts.val_bs, shuffle=False)
    Logger.get_logger().info(
        dict(
            train_size=len(train_dataset),
            val_size=len(val_dataset),
        )
    )
    Logger.get_logger().debug("sample dataset[0]: {0}".format(train_dataset[0]))

    model = TwoTowerRecommenderRankerLitModel(lr=opts.lr, dropout=0.30)
    Logger.get_logger().info("model lightning {0}".format(model))

    logits, rating_logits = model.forward(next(iter(val_dataloader)))
    loss = model.training_step(next(iter(val_dataloader)), 0)
    Logger.get_logger().debug(
        "output_logits={0} , out_size={1} . rating={2} . rating_size={3} , loss={4}".format(
            logits, logits.size(), rating_logits, rating_logits.size(), loss
        )
    )

    Logger.get_logger().debug(
        "the rating argmax = {0} , original rating = {1}".format(
            rating_logits.argmax(dim=1), next(iter(val_dataloader))["rating"]
        )
    )

    # Logger.get_logger().debug(BidirectionEncoderDecoderTransformer(30522, 128, 128, 11))

    ## #######################################
    ## training initiatition step
    # running lightning training module
    ## #######################################

    trainer = pl.Trainer(
        fast_dev_run=False,
        logger=TensorBoardLogger("lightning_logs", name="twotower_model"),
        max_epochs=3,
        log_every_n_steps=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        callbacks=[
            EarlyStopping(monitor="val_auc", verbose=True, min_delta=0),
            ModelCheckpoint(
                verbose=True,
                filename="{0}-{1}".format(opts.modelName, opts.version),
                monitor="val_auc",
                save_weights_only=True,
            ),
            ModelSummary(),
        ],
    )
    Logger.get_logger().info("model training will be initated....")
    trainer.fit(model, train_dataloader, val_dataloader)
    Logger.get_logger().info(
        "model training has been finished... outputs has been saved into the root-directory of current folder...."
    )
