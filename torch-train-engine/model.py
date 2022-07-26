"""
# _* coding: utf8 *_

filename: model.py

@author: sounishnath
createdAt: 2022-07-21 20:39:46
"""

from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE


class Model(nn.Module):
    def __init__(self, n_inputs) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 64)
        # self.out = nn.Linear(64, 1)
        self.out = nn.Linear(64, 5)

    def loss(self, yout, target=None):
        if target is None:
            return None
        # return nn.BCEWithLogitsLoss()(yout, target)
        return nn.CrossEntropyLoss()(yout, target)

    def compute_metrics(self, yout, target=None):
        if target is None:
            return {}

        # yout = torch.sigmoid(yout).detach().cpu().numpy() >= 0.55
        yout = torch.softmax(yout).detach().cpu().numpy().argmax(axis=1)
        target = target.detach().cpu().numpy()

        return {
            "accuracy_score": torch.tensor(
                metrics.accuracy_score(target, yout), device=DEVICE, dtype=torch.float32
            ),
            "f1": torch.tensor(
                metrics.f1_score(target, yout, average="macro"),
                device=DEVICE,
                dtype=torch.float32,
            ),
            "roc_auc": torch.tensor(
                metrics.roc_auc_score(target, yout, average="macro"),
                device=DEVICE,
                dtype=torch.float32,
            ),
        }

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=0.77, gamma=0.20)
        return opt, sch

    def forward(self, data, target):
        x = F.relu(self.fc1(data))
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.dropout(x)
        yout = self.out(x)

        # loss = self.loss(yout.view(-1,1), target.view(-1, 1))
        # met = self.compute_metrics(yout.view(-1,1), target.view(-1, 1))
        # return yout, loss, met
        return yout
