"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2022-06-11 01:21:30
"""

import pandas as pd
import tez
from sklearn import metrics, model_selection

import config
import torch
import torch.nn as nn
import torch.nn.functional as F

# Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


class Dataset:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        return dict(
            data=torch.tensor(data, dtype=torch.float),
            target=torch.tensor(self.target[item], dtype=torch.float),
        )


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, 84)
        self.out = nn.Linear(84, 1)

    def loss(self, yout, target=None):
        if target is None:
            return None
        return nn.BCEWithLogitsLoss()(yout, target)

    def optimizer_scheduler(self):
        opt = torch.optim.SGD(self.parameters(), lr=0.001)
        sch = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=3, gamma=0.77)
        return opt, sch

    def monitor_metrics(self, yout, target):
        target = target.cpu().detach().numpy()
        yout = torch.sigmoid(yout).cpu().detach().numpy() >= 0.50
        accuracy = metrics.accuracy_score(target, yout)
        return dict(
            accuracy=torch.tensor(accuracy, device="cpu"),
        )

    def forward(self, data, target=None):
        out = F.relu6(self.linear(data))
        out = self.out(out)
        loss = self.loss(out, target.view(-1, 1))
        compute_metrics = self.monitor_metrics(out, target.view(-1, 1))
        return out, loss, compute_metrics


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    columns = [
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Class",
    ]
    df = df[columns].reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df["kfold"] = -1
    X = df[columns[:-1]]
    y = df[columns[-1]]
    skf = model_selection.StratifiedKFold(n_splits=10)
    for fold, (trn_, valid_) in enumerate(skf.split(X, y)):
        df.loc[valid_, "kfold"] = fold

    df_train = (
        df[df.kfold != config.FOLD].reset_index(drop=True).drop(["kfold"], axis=1)
    )
    df_valid = (
        df[df.kfold == config.FOLD].reset_index(drop=True).drop(["kfold"], axis=1)
    )

    train_dataset = Dataset(
        df_train.drop(["Class"], axis=1).values, df_train["Class"].values
    )
    valid_dataset = Dataset(
        df_valid.drop(["Class"], axis=1).values, df_valid["Class"].values
    )

    model = Model(input_size=(28))
    model = tez.Tez(model)
    es = tez.callbacks.EarlyStopping(monitor="valid_loss", model_path="model.bin")
    configuration = tez.TezConfig(
        device=config.DEVICE,
        training_batch_size=config.TRAIN_BATCH_SIZE,
        validation_batch_size=config.VALID_BATCH_SIZE,
        epochs=config.EPOCHS,
        gradient_accumulation_steps=1,
        clip_grad_norm=1,
        step_scheduler_after="epoch",
    )
    model.fit(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        config=configuration,
        callbacks=[es],
    )
