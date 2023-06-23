"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-06-23 21:16:48
"""

import os
from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import pytorch_lightning as pl
import shutil
import pandas as pd
import logging

from sklearn import datasets
from sklearn import model_selection

import mlflow.pytorch


def to_log(s):
    return {"log_payload": s}


class FashionMnistDataset:
    def __init__(self, data, target) -> None:
        self._data = data
        self._target = target

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        data = self._data[item]
        target = self._target[item]

        return dict(
            data=torch.tensor(data / 255.0, dtype=torch.float).reshape(1, 28, 28),
            target=torch.tensor(target, dtype=torch.long),
        )


class FashionMnistDeepNeuralNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(FashionMnistDeepNeuralNet, self).__init__(*args, **kwargs)
        logging.basicConfig(level=logging.DEBUG)
        self.convolution_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.25),
        )
        self.convolution_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.25),
        )
        self.convolution_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.25),
        )
        self.classfier = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.20),
            nn.Linear(512, 10),
        )

    def forward(self, data, target=None):
        _logits = F.relu(self.convolution_block_1(data))
        # logging.info(to_log(dict(conv_block_1=_logits.size())))
        _logits = F.relu(self.convolution_block_2(_logits))
        # logging.info(to_log(dict(conv_block_2=_logits.size())))
        _logits = F.relu(self.convolution_block_3(_logits))
        # logging.info(to_log(dict(conv_block_3=_logits.size())))
        _logits = nn.Flatten()(_logits)
        _logits = self.classfier(_logits)
        # logging.info(to_log(dict(_logits=_logits.size())))
        return _logits


class LightningFashionMnistDeepNeuralNet(pl.LightningModule):
    def __init__(self, foundational_model, *args: Any, **kwargs: Any) -> None:
        super(LightningFashionMnistDeepNeuralNet, self).__init__(*args, **kwargs)
        self._foundational_model = foundational_model
        self.save_hyperparameters(ignore=["foundational_model"])

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        data = batch["data"]
        target = batch["target"]
        ypred = self._foundational_model.forward(data)
        loss = nn.CrossEntropyLoss()(ypred, target)
        self.log("train_loss", loss, prog_bar=True, on_step=True)

        _predictions = torch.softmax(ypred, dim=1).argmax(dim=1)
        train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=10,
        )(
            _predictions.detach().cpu(),
            target.detach().cpu(),
        )
        self.log("train_acc", train_acc, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        data = batch["data"]
        target = batch["target"]
        ypred = self._foundational_model.forward(data)
        loss = nn.CrossEntropyLoss()(ypred, target)
        self.log("valid_loss", loss, prog_bar=True)

        _predictions = torch.softmax(ypred, dim=1).argmax(dim=1)
        valid_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=10,
        )(
            _predictions.detach().cpu(),
            target.detach().cpu(),
        )
        self.log("valid_acc", valid_acc, prog_bar=True)

        return {"valid_loss": loss, "valid_acc": valid_acc}

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        data = batch["data"]
        target = batch["target"]
        ypred = self._foundational_model.forward(data)
        loss = nn.CrossEntropyLoss()(ypred, target)
        self.log("test_loss", loss, prog_bar=True)

        _predictions = torch.softmax(ypred, dim=1).argmax(dim=1)
        test_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=10,
        )(
            _predictions.detach().cpu(),
            target.detach().cpu(),
        )

        return {"test_loss": loss, "test_acc": test_acc}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        data = batch["data"]
        ypred = self._foundational_model.forward(data)
        return ypred

    def configure_optimizers(self) -> Any:
        opt = torch.optim.Adam(self.parameters(), lr=3e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode="min",
            factor=0.01,
            patience=3,
            verbose=True,
            min_lr=1e-3,
        )
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "valid_loss"}


if __name__ == "__main__":
    mlflow.pytorch.autolog()
    pl.seed_everything(1010, workers=True)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("output.log", encoding="utf-8", delay=True),
        ],
    )
    logger = logging.getLogger("Torchf")

    if not os.path.exists(os.path.join(os.getcwd(), "data", "mnist_dataset.csv")):
        mnist_dataset = datasets.fetch_openml("mnist_784")
        logger.info(to_log(f"mnist={dir(mnist_dataset)}"))
        logger.info(to_log(mnist_dataset.target_names))
        logger.info(to_log(mnist_dataset.url))
        logger.info(to_log(f"{os.path.join(os.getcwd(), 'data')} path not found!!."))
        os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)
        mnist_dataset_dfx = mnist_dataset.data.copy().iloc[:, :-1]
        mnist_dataset_dfx.loc[:, "target"] = mnist_dataset.target.values
        mnist_dataset_dfx.to_csv(
            os.path.join(os.getcwd(), "data", "mnist_dataset.csv"), index=False
        )
        logger.info(to_log(f"{os.path.join(os.getcwd(), 'data')} has been saved"))

        logger.info(to_log(f"total_size={mnist_dataset_dfx.shape}"))
        logger.info(to_log(f"last 5 cols={mnist_dataset_dfx.columns[-5:]}"))

        mnist_dataset_dfx.loc[:, "kfold"] = -1
        skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
        X = mnist_dataset_dfx.drop("target", axis=1)
        y = mnist_dataset_dfx["target"].values
        for batch, (trn_index, valid_index) in enumerate(skf.split(X, y)):
            mnist_dataset_dfx.loc[valid_index, "kfold"] = batch

        mnist_dataset_dfx.to_csv(
            os.path.join(os.getcwd(), "data", "mnist_dataset_kfold.csv"), index=False
        )
        logger.info(
            to_log(
                f"{os.path.join(os.getcwd(), 'data', 'mnist_dataset_kfold.csv')} 5-folded data has been prepared!"
            )
        )

    else:
        mnist_dataset_dfx = pd.read_csv(
            os.path.join(os.getcwd(), "data", "mnist_dataset_kfold.csv")
        )
        logger.info(
            to_log(f"mnist_dataset_kfold.csv has been loaded from local directory")
        )

    total_rows = mnist_dataset_dfx[mnist_dataset_dfx["kfold"] != 0].shape[0]
    train_size = int(total_rows * 0.88)
    train_dataset = mnist_dataset_dfx[mnist_dataset_dfx["kfold"] != 0][:train_size]
    eval_dataset = mnist_dataset_dfx[mnist_dataset_dfx["kfold"] != 0][train_size:]
    test_dataset = mnist_dataset_dfx[mnist_dataset_dfx["kfold"] == 0]
    logger.info(
        to_log(
            dict(
                trainsize=train_dataset.shape,
                evalsize=eval_dataset.shape,
                testsize=test_dataset.shape,
            )
        )
    )

    train_dataset = FashionMnistDataset(
        data=train_dataset.drop("target", axis=1).values,
        target=train_dataset["target"].values,
    )
    eval_dataset = FashionMnistDataset(
        data=eval_dataset.drop("target", axis=1).values,
        target=eval_dataset["target"].values,
    )
    test_dataset = FashionMnistDataset(
        data=test_dataset.drop("target", axis=1).values,
        target=test_dataset["target"].values,
    )

    model = FashionMnistDeepNeuralNet()

    lit_model = LightningFashionMnistDeepNeuralNet(foundational_model=model)
    print(lit_model)

    trainer = pl.Trainer(
        max_epochs=6,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        fast_dev_run=False,
        detect_anomaly=True,
        enable_model_summary=True,
        # callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=3
        ),
        val_dataloaders=DataLoader(
            eval_dataset, batch_size=16, shuffle=False, num_workers=2
        ),
    )

    torch.save(lit_model, "openml_fashionmnist_minivgg.pt")
