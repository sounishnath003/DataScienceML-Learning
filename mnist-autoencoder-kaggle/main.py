"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-02-24 23:15:06
"""

# pip install torch pytorch_lightning numpy scikit-learn pandas

import logging
from dataclasses import dataclass

import pandas as pd
import pytorch_lightning as lightning
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    Sampler,
    random_split,
)

import torch


class PrintCallback(lightning.callbacks.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


@dataclass
class Config:
    TRAIN_PATH: str = "./data/mnist_train.csv"
    TEST_PATH: str = "./data/mnist_test.csv"
    EPOCHS: int = 3
    TRAIN_BS: int = 8
    VALID_BS: int = 8
    TARGET_COL: str = "label"
    VALID_RATIO: float = 0.20
    LEARNING_RATE: float = 3e-4


class MnistDataset:
    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        target = self.targets[item]

        return {
            "data": torch.tensor(data, dtype=torch.float),
            "target": torch.tensor(target, dtype=torch.long),
        }


class EncoderNetwork(nn.Module):
    def __init__(self) -> None:
        super(EncoderNetwork, self).__init__()
        self.encoder_layer = nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 10),
        )

    def forward(self, data):
        return self.encoder_layer(data)


class DencoderNetwork(nn.Module):
    def __init__(self) -> None:
        super(DencoderNetwork, self).__init__()
        self.decoder_layer = nn.Sequential(
            torch.nn.Linear(10, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid(),
        )

    def forward(self, data):
        return self.decoder_layer(data)


class MnistAutoEncoderNeuralNetLightningModel(lightning.LightningModule):
    def __init__(self, encoder: EncoderNetwork, decoder: DencoderNetwork) -> None:
        super(MnistAutoEncoderNeuralNetLightningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dropout = nn.Dropout(0.11)
        self.classfier = nn.Sequential(
            nn.Linear(28 * 28, 28 * 28),
            nn.Linear(28 * 28, 512),
            nn.Linear(28 * 28, 512),
            nn.Dropout(0.20),
            nn.Linear(512, 10),
            nn.Dropout(0.05),
        )

        self.save_hyperparameters()

    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)

    def configure_optimizers(self):
        opt_params = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
        ]
        opt = torch.optim.Adam(params=opt_params, lr=Config.LEARNING_RATE)
        sch = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=3, gamma=0.20)

        return {"optimizer": opt, "lr_scheduler": sch}

    def __common_step(self, batch):
        outs = self.forward(**batch)
        F.dropout(outs)
        loss = self.loss_fn(outs, batch["target"])
        return outs, loss

    def training_step(self, batch, batch_idx):
        outs, loss = self.__common_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outs, loss = self.__common_step(batch)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        outs, loss = self.__common_step(batch)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        outs, loss = self.__common_step(batch)
        return outs, loss

    def forward(self, data, target):
        enc_out = self.encoder(data)
        enc_out = self.dropout(enc_out)
        dec_out = self.decoder(enc_out)
        dec_out = self.dropout(dec_out)
        logits = self.classfier(dec_out)

        return logits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    logging.info(Config())

    train_dfx = pd.read_csv(Config.TRAIN_PATH)
    test_dfx = pd.read_csv(Config.TEST_PATH)

    logging.info(f"train_dfx: {train_dfx.columns}")
    logging.info(f"train_dfx: {train_dfx.shape}")
    logging.info(f"test_dfx: {test_dfx.shape}")

    feature_cols = [col for col in train_dfx.columns if col != Config.TARGET_COL]
    data_train, targets_train = train_dfx[feature_cols].values, train_dfx.pop(
        Config.TARGET_COL
    )
    data_test, targets_test = test_dfx[feature_cols].values, test_dfx.pop(
        Config.TARGET_COL
    )

    train_dataset = MnistDataset(data=data_train, targets=targets_train)
    test_dataset = MnistDataset(data=data_test, targets=targets_test)

    train_dataset, valid_dataset = random_split(
        dataset=train_dataset,
        lengths=[
            int(len(train_dataset) * (1 - Config.VALID_RATIO)),
            int(len(train_dataset) * (Config.VALID_RATIO)),
        ],
    )
    logging.info(
        dict(
            train_size=len(train_dataset),
            valid_size=len(valid_dataset),
            test_size=len(test_dataset),
        )
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.TRAIN_BS,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=Config.VALID_BS,
        shuffle=False,
        drop_last=True,
        num_workers=8,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=Config.VALID_BS,
        shuffle=False,
        drop_last=True,
        num_workers=8,
    )

    model = MnistAutoEncoderNeuralNetLightningModel(
        encoder=EncoderNetwork(), decoder=DencoderNetwork()
    )
    logging.info(model)

    callbacks = [PrintCallback()]
    trainer = lightning.Trainer(
        accelerator="cpu",
        max_epochs=Config.EPOCHS,
        accumulate_grad_batches=1,
        callbacks=callbacks,
        check_val_every_n_epoch=2,
        precision=16,
    )

    trainer.fit(model, train_dataloader, valid_dataloader)
    
    # preload from the pretrained_checkpoints
    # model=MnistAutoEncoderNeuralNetLightningModel.from_checkpoints('CHECKPOINT_PATH', encoder=encoder, decoder=decoder)
    # basic simple example of the autoencoder models
