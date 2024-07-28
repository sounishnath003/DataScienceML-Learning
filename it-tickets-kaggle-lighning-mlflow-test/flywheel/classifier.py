# * coding utf-8 *
# @author: @github/sounishnath003
# createdAt: 25-07-2024

from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from loguru import logger
import torch
from lightning import pytorch as pl
from torchmetrics import classification
from transformers import DistilBertForSequenceClassification, DistilBertConfig


class TextClassifierModel(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.25,
        attention_dropout: float = 0.20,
        seq_classif_dropout: float = 0.10,
    ) -> None:
        super(TextClassifierModel, self).__init__()
        self.config = DistilBertConfig(
            dropout=dropout,
            attention_dropout=attention_dropout,
            seq_classif_dropout=seq_classif_dropout,
            num_labels=num_classes,
        )
        self.pretrained_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", config=self.config
        )

        self.disable_finetuning_of_pretrained_layers()
        # self.init_random_weights()

    def init_random_weights(self):
        """randomize using xavier uniform to reset and preset with randomized weights for the distilbert model"""
        torch.manual_seed(42)
        for param in self.pretrained_model.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
        logger.debug(
            "Model weights have been randomized using Xavier uniform initialization."
        )

    def disable_finetuning_of_pretrained_layers(self):
        """set requires_grad=False for the pretrained weights of the distilbert model. can keep the head of the transformer traininable i.e. requires_grad=True"""
        for name, param in self.pretrained_model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        logger.debug(
            "Pretrained layers have been frozen, only the classifier head is trainable."
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """returns the logits from the pretrained model"""
        outputs = self.pretrained_model.forward(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        return outputs.logits


class TextClassifierLitModel(pl.LightningModule):
    def __init__(
        self, num_classes: int, lr: float = 3e-4, *args: Any, **kwargs: Any
    ) -> None:
        super(TextClassifierLitModel, self).__init__(*args, **kwargs)
        pl.seed_everything(42)

        self.lr = lr
        self.num_classes = num_classes

        logger.debug("torch random seed has been set to 42")
        self.model = TextClassifierModel(num_classes=num_classes)
        logger.info("model.config: {}", self.model.config)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc_metric = classification.MulticlassAccuracy(
            num_classes=self.num_classes
        ).to(self.device)
        self.f1_metric = classification.MulticlassF1Score(
            num_classes=self.num_classes
        ).to(self.device)

    def training_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, logits = self.forward(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            label=batch.get("labels"),
        )
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # calculate the accuracy
        acc_score = self.acc_metric(
            torch.softmax(logits, dim=1).argmax(dim=1),
            batch["labels"],
        )
        self.log("acc", acc_score, prog_bar=True, on_step=False, on_epoch=True)

        f1_score = self.f1_metric(
            torch.softmax(logits, dim=1).argmax(dim=1),
            batch["labels"],
        )
        self.log("f1", f1_score, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch, batch_id, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        self.eval()
        loss, logits = self.forward(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            label=batch.get("labels"),
        )
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # calculate the accuracy
        acc_score = self.acc_metric(
            torch.softmax(logits, dim=1).argmax(dim=1),
            batch["labels"],
        )
        self.log("val_acc", acc_score, prog_bar=True, on_step=False, on_epoch=True)

        f1_score = self.f1_metric(
            torch.softmax(logits, dim=1).argmax(dim=1),
            batch["labels"],
        )
        self.log("val_f1", f1_score, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            self.eval()
            _, logits = self.forward(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                label=None,
            )

            preds = torch.softmax(logits, dim=1)
            preds.requires_grad = False

            return preds

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(
            params=[
                param
                for param in self.model.parameters()
                if param.requires_grad == True
            ],
            eps=1e-6,
            amsgrad=True,
            lr=self.lr,
        )
        sch = torch.optim.lr_scheduler.StepLR(opt, gamma=0.11, step_size=100)

        return [opt], [sch]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label: torch.Tensor = None,
    ):
        """returns the (loss,logits) from the self.forward layer"""
        loss = 0.0
        logits = self.model.forward(input_ids, attention_mask)

        if label is not None:
            loss = self.loss_fn(logits, label)

        return loss, logits
