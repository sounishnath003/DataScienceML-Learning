"""
# _* coding: utf8 *_

filename: content_preference_model.py

@author: sounishnath
createdAt: 2023-10-28 19:57:53
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from abc import ABC, abstractmethod
from transformers import AutoModel, AutoConfig


class ContentPreferencesModel(nn.Module):
    def __init__(self, pretrainedHfModel: str, dropout: float, *args, **kwargs) -> None:
        super(ContentPreferencesModel, self).__init__(*args, **kwargs)
        self.pretrainedHfModel = pretrainedHfModel
        self.config = AutoConfig.from_pretrained(pretrainedHfModel)
        self.pretrained_model = AutoModel.from_pretrained(pretrainedHfModel)
        self.linear = nn.Linear(self.config.dim, self.config.dim)
        self.dropout = nn.Dropout(0.30)
        self.clf = nn.Linear(self.config.dim, 384)

    def forward(self, content_input_ids, content_attention_mask):
        logits = self.pretrained_model.forward(
            content_input_ids, content_attention_mask
        )[0][:, 0]
        logits = F.relu(self.linear.forward(logits))
        logits = self.dropout.forward(logits)
        return self.clf.forward(logits)
