"""
# _* coding: utf8 *_

filename: inference.py

@author: sounishnath
createdAt: 2023-09-09 15:17:02
"""

import torch
from lightning import pytorch as pl
from recurrent_variation_autoencoder import (
    RecurrentVariationAutoEncoderTimeseriesClusteringLit,
    TimeseriesDataSet,
)
from torch.utils.data import DataLoader


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


if __name__ == "__main__":
    input_np = [
        torch.randn(Configuration.SEQUENCE_LENGTH, Configuration.NUM_OF_FEATURES)
        .detach()
        .cpu()
        .numpy()
        for _ in range(100)
    ]

    lit_model = RecurrentVariationAutoEncoderTimeseriesClusteringLit.load_from_checkpoint(
        "/Users/sounishnath/workspaces/torchf/recurrent-variation-autoencoder/lightning_logs/version_0/checkpoints/epoch=1-step=20.ckpt",
        sequence_length=Configuration.SEQUENCE_LENGTH,
        num_of_features=Configuration.NUM_OF_FEATURES,
        hidden_size=Configuration.HIDDEN_SIZE,
        hidden_layer_depth=Configuration.HIDDEN_LAYER_DEPTH,
        latent_dim=Configuration.LATENT_DIM,
        batch_size=Configuration.BATCH_SIZE,
        learning_rate=Configuration.LEARNING_RATE,
        dropout=Configuration.DROPOUT,
    )
    lit_model.eval()

    trainer = pl.Trainer()
    predict_dataloader = DataLoader(
        dataset=TimeseriesDataSet(input_np),
        batch_size=64,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )
    with torch.no_grad():
        preds = trainer.test(lit_model, predict_dataloader, verbose=True)
        print(preds)
