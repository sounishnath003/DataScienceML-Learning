"""
# _* coding: utf8 *_

filename: twotower_recommender.py

@author: sounishnath
createdAt: 2023-10-28 19:58:55
"""


from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics import AUROC, Dice, Recall

from .logger import Logger
from .models.content_preference_model import ContentPreferencesModel
from .models.preranking_model import LightSqueezeExicitationModel
from .models.user_preference_model import UserPreferencesModel


class TwoTowerRecommenderRankerLitModel(pl.LightningModule):
    def __init__(
        self,
        pretrained_hf: str = "distilbert-base-uncased",
        lr: float = 1e-3,
        dropout: float = 0.30,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super(TwoTowerRecommenderRankerLitModel, self).__init__(*args, **kwargs)
        self.lr = lr
        self.user_preference_model = UserPreferencesModel(pretrained_hf, dropout)
        self.content_preference_model = ContentPreferencesModel(pretrained_hf, dropout)
        self.layer_normalizer = nn.LayerNorm(normalized_shape=(768))
        self.light_se = LightSqueezeExicitationModel(field_size=768)
        self.combiner = nn.Linear(384 * 2, 768)
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Linear(768, 768)
        self.rating_predictor = nn.Linear(768, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        user_input_ids, user_attention_mask = (
            X["user_detail"]["input_ids"],
            X["user_detail"]["attention_mask"],
        )
        content_input_ids, content_attention_mask = (
            X["content_detail"]["input_ids"],
            X["content_detail"]["attention_mask"],
        )

        user_logits = self.user_preference_model.forward(
            user_input_ids, user_attention_mask
        )
        Logger.get_logger().debug("user_logits_size={0}".format(user_logits.size()))
        content_logits = self.content_preference_model.forward(
            content_input_ids, content_attention_mask
        )
        Logger.get_logger().debug(
            "content_logits_size={0}".format(content_logits.size())
        )

        matrix_mul_logits = content_logits @ torch.matmul(user_logits.T, content_logits)
        Logger.get_logger().debug(
            "matrix-multipication = {0} . size={1}".format(
                matrix_mul_logits,
                matrix_mul_logits.size(),
            )
        )

        concated_logits = self.layer_normalizer.forward(
            self.light_se.forward(
                F.relu(
                    self.combiner.forward(
                        torch.concat([user_logits, matrix_mul_logits], dim=-1)
                    )
                )
            )
        )
        concated_logits = self.dropout.forward(concated_logits)

        rating_concated_logits = F.relu(
            self.rating_predictor.forward(
                self.light_se.forward(
                    torch.concat([user_logits, matrix_mul_logits], dim=-1)
                )
            )
        )
        Logger.get_logger().debug(
            "concated_logits_size={0}".format(concated_logits.size())
        )

        logits = torch.tensor(100.0) * self.softmax.forward(
            self.classifier.forward(concated_logits)
        )
        Logger.get_logger().debug("classifier_logits_size={0}".format(logits.size()))
        rating_logits = torch.tensor(1.0) * rating_concated_logits
        return (logits, rating_logits)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer=opt, verbose=True, step_size=30, gamma=0.1
        )
        return [opt], [sch]

    def compute_loss(self, rating_logits, original_rating):
        loss = nn.CrossEntropyLoss()(rating_logits, original_rating)
        return loss

    def compute_metric(self, rating_logits, original_rating, eval=False):
        rating_logits = rating_logits.cpu().detach()
        original_rating = original_rating.cpu().detach()

        aucroc_score = AUROC(task="multiclass", num_classes=5)(
            rating_logits, original_rating
        )
        recall_score = Recall(task="multiclass", num_classes=5)(
            rating_logits, original_rating
        )
        dice_score = Dice(average="micro")(rating_logits, original_rating)

        scores = {}
        if eval:
            scores = dict(val_auc=aucroc_score, val_recall=recall_score)
        else:
            scores = dict(auc=aucroc_score, recall=recall_score)
        return scores

    def training_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        logits, rating_logits = self.forward(batch)
        loss = self.compute_loss(rating_logits, batch["rating"])
        scores_metrics = self.compute_metric(rating_logits, batch["rating"])
        self.log_dict(
            dictionary=dict(loss=loss), prog_bar=True, on_step=True, on_epoch=True
        )
        self.log_dict(
            dictionary=scores_metrics, prog_bar=True, on_step=False, on_epoch=True
        )
        return loss

    def predict_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        logits, rating_logits = self.forward(batch)
        return dict(embeddings=logits, rating=rating_logits)

    def validation_step(
        self, batch, batch_id, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        logits, rating_logits = self.forward(batch)
        loss = self.compute_loss(rating_logits, batch["rating"])
        self.log_dict(
            dictionary=dict(val_loss=loss), prog_bar=True, on_step=False, on_epoch=True
        )
        scores_metrics = self.compute_metric(rating_logits, batch["rating"], eval=True)
        self.log_dict(
            dictionary=scores_metrics, prog_bar=True, on_step=True, on_epoch=True
        )
        return loss
