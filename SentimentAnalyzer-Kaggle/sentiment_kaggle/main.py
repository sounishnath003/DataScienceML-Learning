"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-06-25 11:38:20
"""

import os
import re
import logging
from typing import Any, Optional, Sequence, Union
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
import pandas as pd
from sklearn import model_selection

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForPreTraining
import torch.nn as nn
import torch.nn.functional as F
from lightning import pytorch as pl
from lightning.pytorch import callbacks
import torchmetrics
from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class Config:
    FAST_DEV_RUN = False
    TRAIN_DATASET = os.path.join(os.getcwd(), "data", "train.tsv")
    TEST_DATASET = os.path.join(os.getcwd(), "data", "test.tsv")
    COLUMN_X = "Phrase"
    COLUMN_Y = "Sentiment"
    HF_MODEL = "distilbert-base-uncased"
    BATCH_SIZE = 32
    MAX_LENGTH = 32
    NUM_CLASSES = 5
    NUM_EPOCHS = 1
    LEARNING_RATE = 3e-5
    NUM_WORKERS = os.cpu_count()
    TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def describe_dfx(dfx):
    logger = logging.getLogger("SentimentAnalysisDatasetLit")
    logger.info(10 * "========")
    logger.info("shape: {0}".format(dfx.shape))
    logger.info("columns: {0}".format(dfx.columns))
    logger.info("sample:\n {0}".format(dfx.sample(5)))
    logger.info(10 * "========")
    logger.info("")


class SentimentAnalysisDataset:
    def __init__(self, data, target) -> None:
        self._data = data
        self._target = target

    def __len__(self):
        return len(self._data)

    def _preprocess_text(self, text: str) -> str:
        return " ".join([word for word in re.split("\W+", text)])

    def __getitem__(self, item):
        data = self._preprocess_text(self._data[item])
        target = self._target[item]
        tokenized_output = Config.TOKENIZER.encode_plus(
            data,
            data,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
            is_split_into_words=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_offsets_mapping=False,
            return_length=False,
            verbose=True,
        )
        return dict(
            input_ids=tokenized_output["input_ids"].squeeze(0),
            attention_mask=tokenized_output["attention_mask"].squeeze(0),
            token_type_ids=tokenized_output["token_type_ids"].squeeze(0),
            target=torch.tensor(target, dtype=torch.long),
        )


class SentimentAnalysisDataLoaderLightningModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32) -> None:
        super(SentimentAnalysisDataLoaderLightningModule).__init__()
        self._batch_size = batch_size

    def _split_into_train_eval_dataset(
        self, _train_dfx: pd.DataFrame, n_splits: int = 5, fold: int = 0
    ):
        skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
        _train_dfx.loc[:, "kfold"] = -1
        for fold, (trn_index_, valid_index_) in enumerate(
            skf.split(
                X=_train_dfx[Config.COLUMN_X].values,
                y=_train_dfx[Config.COLUMN_Y].values,
            )
        ):
            _train_dfx.loc[valid_index_, "kfold"] = fold

        trn_dfx = _train_dfx[_train_dfx["kfold"] != fold]
        valid_dfx = _train_dfx[_train_dfx["kfold"] == fold]

        describe_dfx(trn_dfx)
        describe_dfx(valid_dfx)
        return trn_dfx.copy(), valid_dfx.copy()

    def setup(self, stage: str) -> None:
        self._train_dfx, self._valid_dfx = self._split_into_train_eval_dataset(
            pd.read_csv(Config.TRAIN_DATASET, delimiter="\t")
        )
        self._test_dfx = pd.read_csv(Config.TEST_DATASET, delimiter="\t")
        describe_dfx(self._test_dfx)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=SentimentAnalysisDataset(
                data=self._train_dfx[Config.COLUMN_X].values,
                target=self._train_dfx[Config.COLUMN_Y].values,
            ),
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )

    def _get_random_sample(self):
        return SentimentAnalysisDataset(
            data=self._train_dfx[Config.COLUMN_X].values,
            target=self._train_dfx[Config.COLUMN_Y].values,
        )[2]

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=SentimentAnalysisDataset(
                data=self._valid_dfx[Config.COLUMN_X].values,
                target=self._valid_dfx[Config.COLUMN_Y].values,
            ),
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=SentimentAnalysisDataset(
                data=self._test_dfx[Config.COLUMN_X].values,
                target=self._test_dfx[Config.COLUMN_Y].values,
            ),
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )


class SentimentAnalyzerDeepNeuralNet(nn.Module):
    def __init__(
        self,
        n_classes: int = Config.NUM_CLASSES,
        learning_rate: float = 3e-4,
        *args,
        **kwargs
    ) -> None:
        super(SentimentAnalyzerDeepNeuralNet, self).__init__(*args, **kwargs)
        self._logger = logging.getLogger("SentimentAnalyzerDeepNeuralNet")

        # self.foundation_model = AutoModel.from_pretrained(Config.HF_MODEL)
        self.foundation_model = AutoModelForPreTraining.from_pretrained(Config.HF_MODEL)
        for param in self.foundation_model.base_model.parameters():
            param.requires_grad = False

        self.sentiment_classifier_layer = nn.Sequential(
            nn.Sequential(
                nn.Dropout(0.25),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Linear(15261, 2048),
                nn.Dropout(0.25),
                nn.Linear(2048, 2048),
                nn.LayerNorm(2048),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Dropout(0.25),
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(8192, 2048),
                nn.Dropout(0.25),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, n_classes),
            ),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, target=None):
        _logits = self.foundation_model(
            input_ids,
            attention_mask=attention_mask,
        ).logits
        # logger.info(_logits.size())
        _logits = self.sentiment_classifier_layer(_logits)
        # logger.info(_logits.size())
        return _logits


class SentimentAnalyzerDeepNeuralNetLightning(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = Config.NUM_CLASSES,
        learning_rate: float = Config.LEARNING_RATE,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super(SentimentAnalyzerDeepNeuralNetLightning, self).__init__(*args, **kwargs)
        self._num_classes = num_classes
        self._learning_rate = learning_rate
        self.model = SentimentAnalyzerDeepNeuralNet(
            n_classes=num_classes, learning_rate=self._learning_rate
        )

    def _common_steps(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        target = batch.get("target", None)

        ypreds = self.model.forward(input_ids, attention_mask, token_type_ids, target)
        loss = torch.tensor(0, dtype=torch.float)
        accuracy = torch.tensor(0, dtype=torch.float)
        f1 = torch.tensor(0, dtype=torch.float)

        if not target is None:
            loss = nn.CrossEntropyLoss()(ypreds, target)
            accuracy = torchmetrics.Accuracy(
                task="multiclass", threshold=0.67, num_classes=self._num_classes
            )(ypreds.detach().cpu().argmax(dim=1), target.detach().cpu())
            f1 = torchmetrics.F1Score(
                task="multiclass",
                threshold=0.67,
                num_classes=self._num_classes,
                average="macro",
            )(ypreds.detach().cpu().argmax(dim=1), target.detach().cpu())

        return (
            ypreds,
            loss,
            [{"type": "accuracy", "score": accuracy}, {"type": "f1", "score": f1}],
        )

    def training_step(
        self, batch, batch_index, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        ypreds, loss, metrics = self._common_steps(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        for metric in metrics:
            self.log(
                metric["type"],
                metric["score"],
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )

        return {"predictions": ypreds, "loss": loss}

    def validation_step(
        self, batch, batch_index, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT | None:
        ypreds, loss, metrics = self._common_steps(batch)
        self.log("valid_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        for metric in metrics:
            self.log(
                metric["type"],
                metric["score"],
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )

        return {"predictions": ypreds, "loss": loss}

    def test_step(
        self, batch, batch_index, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT | None:
        ypreds, loss, metrics = self._common_steps(batch)
        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        for metric in metrics:
            self.log(
                metric["type"],
                metric["score"],
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )

        return {"predictions": ypreds, "loss": loss}

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(
            self.parameters(), lr=self._learning_rate, eps=1e-5, weight_decay=0.02
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt, mode="min", factor=0.02, patience=3, min_lr=1e-2, eps=1e-5
        )
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "valid_loss"}

    def forward(self, batch, batch_index, *args: Any, **kwargs: Any) -> Any:
        ypreds, loss, metrics = self._common_steps(batch)
        return {"ypreds": ypreds, "loss": loss}


if __name__ == "__main__":
    pl.seed_everything(101)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("runs.log", encoding="utf-8", delay=True),
        ],
    )
    logger = logging.getLogger("SentimentAnalysisLightning")

    model = SentimentAnalyzerDeepNeuralNet()
    lit_model = SentimentAnalyzerDeepNeuralNetLightning(
        num_classes=Config.NUM_CLASSES, learning_rate=Config.LEARNING_RATE
    )
    logger.info(lit_model)

    trainer = pl.Trainer(
        fast_dev_run=Config.FAST_DEV_RUN,
        max_epochs=Config.NUM_EPOCHS,
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        callbacks=[callbacks.EarlyStopping(monitor="valid_loss", mode="min")],
    )

    def _split_into_train_eval_dataset(
        _train_dfx: pd.DataFrame, n_splits: int = 5, fold: int = 0
    ):
        skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
        _train_dfx.loc[:, "kfold"] = -1
        for fold, (trn_index_, valid_index_) in enumerate(
            skf.split(
                X=_train_dfx[Config.COLUMN_X].values,
                y=_train_dfx[Config.COLUMN_Y].values,
            )
        ):
            _train_dfx.loc[valid_index_, "kfold"] = fold

        trn_dfx = _train_dfx[_train_dfx["kfold"] != fold]
        valid_dfx = _train_dfx[_train_dfx["kfold"] == fold]

        describe_dfx(trn_dfx)
        describe_dfx(valid_dfx)
        return trn_dfx.copy(), valid_dfx.copy()

    _train_dfx, _valid_dfx = _split_into_train_eval_dataset(
        pd.read_csv(Config.TRAIN_DATASET, delimiter="\t")
    )
    _test_dfx = pd.read_csv(Config.TEST_DATASET, delimiter="\t")
    describe_dfx(_test_dfx)

    trainer.fit(
        lit_model,
        train_dataloaders=DataLoader(
            dataset=SentimentAnalysisDataset(
                data=_train_dfx[Config.COLUMN_X].values,
                target=_train_dfx[Config.COLUMN_Y].values,
            ),
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        ),
        val_dataloaders=DataLoader(
            dataset=SentimentAnalysisDataset(
                data=_valid_dfx[Config.COLUMN_X].values,
                target=_valid_dfx[Config.COLUMN_Y].values,
            ),
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        ),
    )

    logging.info(
        trainer.test(
            lit_model,
            DataLoader(
                dataset=SentimentAnalysisDataset(
                    data=_valid_dfx[Config.COLUMN_X].values,
                    target=_valid_dfx[Config.COLUMN_Y].values,
                ),
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            ),
        )
    )
