"""
# _* coding: utf8 *_

filename: model.py

@author: sounishnath
createdAt: 2023-05-23 00:47:30
"""

from typing import Any, List, Optional, Union

import pytorch_lightning as pl
import transformers
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRSchedulerPLType
from transformers import AutoConfig, AutoModel

import torch
import torch.nn as nn


class ImdbNeuralNet(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", *args, **kwargs) -> None:
        super(ImdbNeuralNet, self).__init__()
        self._model_name = model_name

        self._dropout = nn.Dropout(0.20)
        self._foundation_model_config = AutoConfig.from_pretrained(self._model_name)
        print("_foundation_model_config=", self._foundation_model_config)

        self._foundation_model_config.update(
            {
                "_name_or_path": "bert-base-uncased",
                "architectures": ["BertForMaskedLM"],
                "attention_probs_dropout_prob": 0.1,
                "gradient_checkpointing": False,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 768,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 512,
                "model_type": "bert",
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "pad_token_id": 0,
                "position_embedding_type": "absolute",
                "transformers_version": "4.29.2",
                "type_vocab_size": 2,
                "use_cache": True,
                "vocab_size": 30522,
            }
        )
        self._foundation_model = AutoModel.from_pretrained(
            self._model_name, config=self._foundation_model_config
        )

    def forward(self, ids, attention_mask, token_type_ids):
        _logits = self._foundation_model.forward(
            ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
        )
        _logits = self._dropout(_logits.last_hidden_state)
        return _logits[:, 0, :]


class LitImdbNeuralNet(pl.LightningModule):
    def __init__(
        self,
        foundation_model: ImdbNeuralNet,
        n_classes: int,
        learning_rate: float = 3e-4,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super(LitImdbNeuralNet, self).__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.n_classes = n_classes

        self._foundation_model = foundation_model
        self._classifier = nn.Linear(768, n_classes)

        self.save_hyperparameters(ignore=["foundation_model"])

    def forward(self, batch, *args: Any, **kwargs: Any) -> Any:
        _ids = batch["input_ids"]
        _token_type_ids = batch["token_type_ids"]
        _attention_mask = batch["attention_mask"]

        _logits = self._foundation_model.forward(
            ids=_ids, attention_mask=_attention_mask, token_type_ids=_token_type_ids
        )
        return self._classifier(_logits)

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(
            params=self.parameters(), lr=self.learning_rate, eps=1e-5
        )
        # sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode="min")
        sch = transformers.get_linear_schedule_with_warmup(
            optimizer=opt, num_warmup_steps=1, num_training_steps=1000
        )
        return {"optimizer": opt, "lr_scheduler": sch}

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        target = batch["sentiment"]

        _logits = self.forward(batch=batch)
        _loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.ones([self.n_classes], device=self.device)
        )(_logits, target.reshape(-1, 1))

        # calculate acc
        labels_hat = torch.argmax(_logits, dim=1) >= 0.67
        train_acc = torch.sum(target == labels_hat).item() / (len(target) * 1.0)

        self.log("train_loss", _loss, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True)

        return {"loss": _loss}

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        target = batch["sentiment"]

        _logits = self.forward(batch=batch)
        _loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.ones([self.n_classes], device=self.device)
        )(_logits, target.reshape(-1, 1))

        # calculate acc
        labels_hat = torch.argmax(_logits, dim=1) >= 0.67
        val_acc = torch.sum(target == labels_hat).item() / (len(target) * 1.0)

        self.log("valid_loss", _loss, prog_bar=True)
        self.log("valid_acc", val_acc, prog_bar=True)

        return {"loss": _loss, "val_acc": val_acc}
