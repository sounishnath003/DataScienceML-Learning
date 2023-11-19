from typing import Any
from lightning import pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers import PreTrainedModel

from torchmetrics import Accuracy, F1Score


class AlmostAnyTextClassifierLitModel(pl.LightningModule):
    def __init__(
        self,
        model: PreTrainedModel,
        num_classes: int,
        dropout: float = 0.30,
        learning_rate: float = 5e-5,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super(AlmostAnyTextClassifierLitModel, self).__init__(*args, **kwargs)
        self.dropout_val = dropout
        self.learning_rate_val = learning_rate

        self.pretrained_model = model
        self.dropout = nn.Dropout(dropout)

        self.train_score_metrics = [
            ("acc", Accuracy(task="multiclass", num_classes=num_classes)),
            ("f1", F1Score(task="multiclass", num_classes=num_classes)),
        ]
        self.val_score_metrics = [
            ("acc", Accuracy(task="multiclass", num_classes=num_classes)),
            ("f1", F1Score(task="multiclass", num_classes=num_classes)),
        ]
        self.test_score_metrics = [
            ("acc", Accuracy(task="multiclass", num_classes=num_classes)),
            ("f1", F1Score(task="multiclass", num_classes=num_classes)),
        ]

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.pretrained_model.forward(
            input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def compute_metrics(self, predicted, original, mtype):
        with torch.no_grad():
            logits = predicted["logits"].cpu().detach()
            predicted_labels = torch.argmax(logits, dim=1)
            score_metrics = {}

            for metric_key, metric_fn in self.train_score_metrics:
                score_metrics["{0}_{1}".format(mtype, metric_key)] = metric_fn(
                    predicted_labels, original.cpu().detach()
                )

            return score_metrics

    def training_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        self.log(
            "train_loss", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True
        )
        score_metrics = self.compute_metrics(outputs, batch["label"], "train")
        self.log_dict(score_metrics, on_epoch=True, on_step=False)
        return outputs["loss"]

    def validation_step(
        self, batch, batch_id, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        self.log(
            "val_loss", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True
        )
        score_metrics = self.compute_metrics(outputs, batch["label"], "val")
        self.log_dict(score_metrics, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        self.log("test_loss", outputs["loss"], prog_bar=True)
        score_metrics = self.compute_metrics(outputs, batch["label"], "test")
        self.log_dict(score_metrics, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(
            self.trainer.model.parameters(), lr=self.learning_rate_val
        )
        return opt
