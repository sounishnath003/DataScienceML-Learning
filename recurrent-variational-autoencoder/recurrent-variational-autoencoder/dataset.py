"""
# _* coding: utf8 *_

filename: dataset.py

@author: sounishnath
createdAt: 2023-09-07 23:16:13
"""

import torch
from lightning import pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split


class TimeseriesDataSet:
    def __init__(self, series) -> None:
        self.series = series

    def __len__(self):
        return len(self.series)

    def __getitem__(self, item):
        return {
            "timeseries_sequence_sample": torch.tensor(
                self.series[item], dtype=torch.float
            )
        }


class TimeseriesLightningDataModule(pl.LightningDataModule):
    def __init__(self, series) -> None:
        self.series = series
        self.prepare_data()

    def prepare_data(self) -> None:
        train_, valid_ = random_split(
            TimeseriesDataSet(self.series), lengths=[0.7, 0.3]
        )
        self.train = train_
        self.valid = valid_

    def setup(self, stage: str) -> None:
        self.prepare_data()
        self.train_dataset = TimeseriesDataSet(self.train)
        self.valid_dataset = TimeseriesDataSet(self.valid)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=64, shuffle=False, num_workers=2
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.valid_dataset, batch_size=64, shuffle=False, num_workers=2
        )
