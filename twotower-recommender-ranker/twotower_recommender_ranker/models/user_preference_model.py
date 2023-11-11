"""
# _* coding: utf8 *_

filename: user_preference_model.py

@author: sounishnath
createdAt: 2023-10-28 19:58:02
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from transformers import AutoConfig, AutoModel


class UserPreferencesModel(nn.Module):
    def __init__(self, pretrainedHfModel: str, dropout: float, *args, **kwargs) -> None:
        super(UserPreferencesModel, self).__init__(*args, **kwargs)
        self.pretrainedHfModel = pretrainedHfModel
        self.config = AutoConfig.from_pretrained(pretrainedHfModel)
        self.pretrained_model = AutoModel.from_pretrained(pretrainedHfModel)
        self.linear = nn.Linear(self.config.dim, self.config.dim)
        self.dropout = nn.Dropout(0.30)
        self.clf = nn.Linear(self.config.dim, 384)

    def forward(self, user_input_ids, user_attention_mask):
        logits = self.pretrained_model(user_input_ids, user_attention_mask)[0][:, 0]
        logits = F.relu(self.linear.forward(logits))
        logits = self.dropout.forward(logits)
        return self.clf.forward(logits)
