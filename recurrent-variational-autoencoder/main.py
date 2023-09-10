"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-09-09 11:43:20
"""

import logging
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import pytorch as pl
from recurrent_variation_autoencoder import (
    RecurrentVariationAutoEncoderTimeseriesClusteringLit,
    TimeseriesDataSet,
)
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":
    pl.seed_everything(101)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("[RecurrentVariationAutoEncoderModel]")

    @dataclass
    class Configuration:
        TOTAL_SAMPLES: int = 1000
        NUM_OF_FEATURES: int = 1
        SEQUENCE_LENGTH: int = 24
        HIDDEN_SIZE: int = int(SEQUENCE_LENGTH * 0.30) or 2
        HIDDEN_LAYER_DEPTH: int = 2
        LATENT_DIM: int = 128
        DROPOUT: float = 0.20
        LEARNING_RATE: float = 1e-2
        BATCH_SIZE: int = 64
        EPOCHS: int = 2

    logger.info("execution of the model.py started...")
    logger.info(Configuration().__dict__)

    lit_model = RecurrentVariationAutoEncoderTimeseriesClusteringLit(
        sequence_length=Configuration.SEQUENCE_LENGTH,
        num_of_features=Configuration.NUM_OF_FEATURES,
        hidden_size=Configuration.HIDDEN_SIZE,
        hidden_layer_depth=Configuration.HIDDEN_LAYER_DEPTH,
        latent_dim=Configuration.LATENT_DIM,
        batch_size=Configuration.BATCH_SIZE,
        learning_rate=Configuration.LEARNING_RATE,
        dropout=Configuration.DROPOUT,
    )
    logger.info(lit_model)

    input_np = [
        torch.randn(Configuration.SEQUENCE_LENGTH, Configuration.NUM_OF_FEATURES)
        .detach()
        .cpu()
        .numpy()
        for _ in range(Configuration.TOTAL_SAMPLES)
    ]

    dataset = TimeseriesDataSet(input_np)
    train_dataset, valid_dataset = random_split(dataset, lengths=[0.7, 0.3])
    logger.info(
        "sample dataset[item] size = {}".format(
            dataset[4]["timeseries_sequence_sample"].size()
        )
    )
    logging.info(
        "train_dataset={} ; valid_dataset={} ;".format(
            len(train_dataset), len(valid_dataset)
        )
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=os.cpu_count() - 1,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=True,
        num_workers=os.cpu_count() - 1,
    )

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=Configuration.EPOCHS,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
    )

    try:
        trainer.fit(
            model=lit_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )
    except Exception as e:
        logging.error(
            """
*************************************************************
################## An Error Has Occured ##################

Error Caused: {}

################## An Error Has Occured ##################
*************************************************************
""".format(
                e
            )
        )
