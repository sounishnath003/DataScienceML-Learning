import re
import os
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lightning.pytorch import callbacks, loggers
import torch
import argparse
import dataclasses
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from pprint import pformat
from loguru import logger
from datasets import Dataset, load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertConfig
from lightning import pytorch as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


@dataclasses.dataclass
class OptedDataset:
    train_dataset: Dataset
    val_dataset: Dataset
    train_size: int
    val_size: int
    num_classes: int


@dataclasses.dataclass
class TextDatasetOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor


def download_twitter_dataset_from_internet():
    """
    download_twitter_dataset_from_internet
    @returns OptedDataset object contains the referred variables
    """
    dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)

    train_dataset = dataset.get("train")
    val_dataset = dataset.get("test")
    num_classes = len(set([data["label"] for data in train_dataset]))

    return OptedDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        num_classes=num_classes,
    )


class TextDataset:
    def __init__(self, dataset: Dataset, max_length: int = 256) -> None:
        super(TextDataset, self).__init__()
        self.dataset = dataset
        self.num_rows = len(dataset)
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )

    def __len__(self):
        return self.num_rows

    def __getitem__(self, item):
        data = " ".join(re.split(r"\s+", self.dataset[item]["text"]))
        label = self.dataset[item]["label"]

        tokenized = self.tokenizer(
            data,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors=None,
        )

        return dict(
            input_ids=torch.tensor(tokenized["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(tokenized["attention_mask"], dtype=torch.long),
            label=torch.tensor(label, dtype=torch.long),
        )


class TwitterEmotionLightningModel(pl.LightningModule):
    def __init__(
        self, num_classes: int, lr: float = 2e-5, *args: Any, **kwargs: Any
    ) -> None:
        super(TwitterEmotionLightningModel, self).__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.lr = lr
        self.distilbert_config = DistilBertConfig()

        self.model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", config=self.distilbert_config
        )
        # !! freezing all the pretrained layers weights n bias !!
        for param in self.model.parameters():
          param.requires_grad=False

        self.pre_classifier = nn.Linear(
            self.distilbert_config.dim, self.distilbert_config.dim
        )
        self.dropout = nn.Dropout(0.20)
        self.classifier = nn.Linear(self.distilbert_config.dim, num_classes)

        self.save_hyperparameters(self.num_classes, self.distilbert_config, self.lr)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        model_out = self.model.forward(input_ids, attention_mask)
        logits = model_out.last_hidden_state[:, 0, :]
        logits = self.pre_classifier.forward(logits)
        logits = self.dropout.forward(logits)
        logits = self.classifier.forward(logits)

        return logits

    def compute_loss(self, preds, labels):
        loss = nn.CrossEntropyLoss()(preds, labels)
        return loss

    @torch.no_grad()
    def compute_metrics(self, preds, labels, metric_type="train"):
        acc=MulticlassAccuracy(num_classes=self.num_classes)
        f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")

        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        if metric_type == "train":
          metrics_dict = {
              f"acc": acc(preds, labels).item(),
              f"f1": f1(preds, labels).item(),
          }
        else:
          metrics_dict = {
              f"{metric_type}_acc": acc(preds, labels).item(),
              f"{metric_type}_f1": f1(preds, labels).item(),
          }

        # logger.debug(preds.size())
        # logger.debug(labels.size())
        # logger.debug(metrics_dict)

        return metrics_dict

    def training_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("label")

        logits = self.forward(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        metrics = self.compute_metrics(logits, labels, "train")
        self.log_dict(
            metrics, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("label")

        logits = self.forward(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        metrics = self.compute_metrics(logits, labels, "test")
        self.log_dict(
            metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("label")

        logits = self.forward(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        metrics = self.compute_metrics(logits, labels, "val")
        self.log_dict(
            metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("label", None)

        logits = self.forward(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        metrics = self.compute_metrics(logits, labels, "val")
        self.log_dict(
            metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(
            params=self.trainer.model.parameters(),
            lr=self.lr,
            weight_decay=1e-2,
        )
        sch = torch.optim.lr_scheduler.StepLR(
            opt, step_size=30, verbose=True, gamma=1e-1
        )
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val_loss"}


if __name__ == "__main__":
    logger.debug("running the script from src/main.py")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--learningRate", type=float, default=2e-5)

    opts, pipeline_opts = parser.parse_known_args()
    logger.info(pformat(pformat(dict(opts=opts, pipeline_opts=pipeline_opts))))

    loaded_dataset = download_twitter_dataset_from_internet()
    logger.info(
        pformat(
            dict(
                t=loaded_dataset.train_dataset,
                train_size=loaded_dataset.train_size,
                val_size=loaded_dataset.val_size,
                num_classes=loaded_dataset.num_classes,
            )
        )
    )

    train_dataset = TextDataset(loaded_dataset.train_dataset)
    val_dataset = TextDataset(loaded_dataset.val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opts.batchSize,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opts.batchSize,
        shuffle=False,
        num_workers=2,
        drop_last=True,
    )

    lit_model = TwitterEmotionLightningModel(num_classes=loaded_dataset.num_classes)
    logger.info(lit_model)

    callbacks = [
        callbacks.ModelSummary(),
        callbacks.EarlyStopping(
            monitor="val_f1", mode="max", verbose=True, check_on_train_epoch_end=True
        ),
        callbacks.ModelCheckpoint(
            dirpath="lightning_logs",
            verbose=True,
            save_top_k=1,
            save_weights_only=True,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=10,
        fast_dev_run=False,
        logger=loggers.TensorBoardLogger("lightning_logs", "models-twitter-emotion-classifier"),
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    trainer.fit(
        lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    test_metrics = pl.Trainer().test(lit_model, val_dataloader)
    logger.info(f"test_metrics={test_metrics}")
