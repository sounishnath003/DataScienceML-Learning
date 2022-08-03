"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2022-08-03 19:30:25
"""

from dataclasses import dataclass
from matplotlib import pyplot as plt

import seaborn as sns
import pandas as pd
from numpy import dtype
from sklearn import metrics, model_selection, preprocessing

import torch
import torch.nn as nn
import torch.utils.data

# Dataset: https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset
# KaggleSolution: https://www.kaggle.com/code/sounishnath003/webpage-phishin-detection-pytorch

@dataclass
class Config:
    DATASET = "dataset_phishing.csv"
    TARGET_COL = "status"
    DEVICE = "mps"
    EPOCHS = 10
    TRAIN_BS = 1024
    VALID_BS = 1024
    LR = 3e-3


class Dataset:
    def __init__(self, data, target) -> None:
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return dict(
            data=torch.tensor(self.data[item], dtype=torch.float),
            target=torch.tensor(self.target[item], dtype=torch.float),
        )


class Model(nn.Module):
    def __init__(self, n_inputs, hidden_units_1, hidden_units_2, n_outputs) -> None:
        super().__init__()
        self.pipeline1 = nn.Sequential(
            nn.Linear(n_inputs, hidden_units_1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units_1),
            nn.Linear(hidden_units_1, hidden_units_1),
            nn.BatchNorm1d(hidden_units_1),
            nn.Dropout(0.20),
        )
        self.pipeline2 = nn.Sequential(
            nn.Linear(hidden_units_1, hidden_units_2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units_2),
            nn.Linear(hidden_units_2, hidden_units_2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units_2),
            nn.Dropout(0.20),
            nn.Linear(hidden_units_2, hidden_units_1),
        )
        self.output_pipeline = nn.Sequential(nn.Linear(hidden_units_1, n_outputs))

    def forward(self, data):
        out = self.pipeline1(data)
        out = self.pipeline2(out)
        out = self.output_pipeline(out)
        return out


class Engine:
    @staticmethod
    def train(epoch, model, dataset, criterion, optimizer, scheduler):
        model.to(Config.DEVICE)
        model.train()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=Config.TRAIN_BS)
        losses = 0
        for batch_idx, tdata in enumerate(dataloader):
            data = tdata["data"].to(Config.DEVICE)
            target = tdata["target"].to(Config.DEVICE)
            output = model(data)
            optimizer.zero_grad()
            loss = criterion(output, target.view(-1, 1))
            com_metric = Engine.compute_metrics(output, target.view(-1, 1))
            losses += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            print(
                "[train]: epoch:{0} batch:[{1}/{2}] loss:{3} metrics:{4} ".format(
                    epoch,
                    batch_idx,
                    len(dataloader),
                    (losses / len(dataloader)),
                    com_metric,
                )
            )

    @staticmethod
    def eval(epoch, model, dataset):
        print("---------- * --------------\n")
        model.to(Config.DEVICE)
        model.eval()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=Config.VALID_BS)
        with torch.no_grad():
            for batch_idx, tdata in enumerate(dataloader):
                data = tdata["data"].to(Config.DEVICE)
                target = tdata["target"].to(Config.DEVICE)
                output = model(data)
                com_metric = Engine.compute_metrics(output, target.view(-1, 1))

                print(
                    "[valid]: epoch:{0} batch:[{1}/{2}] metrics:{3} ".format(
                        epoch,
                        batch_idx,
                        len(dataloader),
                        com_metric,
                    )
                )
            print("---------- * --------------\n")

    @staticmethod
    def generate_predictions(model, dataset, batch_size=1024):
        predictions = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            model.eval()
            for batch_idx, tdata in enumerate(dataloader):
                data = tdata["data"].to(Config.DEVICE)
                target = tdata["target"].to(Config.DEVICE)
                output = torch.sigmoid(model(data)).detach().cpu().numpy() >= 0.5
                predictions.extend(output)

        return predictions

    @staticmethod
    def compute_metrics(output, target=None):
        if target is None:
            return {}

        device = target.device.type
        output = torch.sigmoid(output).detach().cpu().numpy() >= 0.5
        target = target.detach().cpu().numpy()

        accuracy = metrics.accuracy_score(target, output)
        f1_score = metrics.f1_score(target, output)
        return dict(
            accuracy=torch.tensor(accuracy, device=device, dtype=torch.float32),
            f1_score=torch.tensor(f1_score, device=device, dtype=torch.float32),
        )


if __name__ == "__main__":
    dfx = pd.read_csv(Config.DATASET)
    cat_cols = [col for col in dfx.columns if dfx[col].dtype == "object"]
    print("cat_cols:", cat_cols)
    dfx["target"] = preprocessing.LabelEncoder().fit_transform(dfx[Config.TARGET_COL])
    print(dfx[[*cat_cols, "target"]].sample(5))
    df = dfx.drop(["url", "status"], axis=1)

    # corr = df.corr()
    # plt.imshow(corr, cmap="viridis", interpolation="nearest")
    # plt.colorbar()
    # plt.show()

    data = preprocessing.MinMaxScaler().fit_transform(
        df.drop(["target"], axis=1).values
    )
    targets = df["target"].values

    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(
        data, targets, test_size=0.20, random_state=0, stratify=df["target"]
    )
    print(
        dict(
            xtrain_shape=xtrain.shape,
            xtest_shape=xtest.shape,
        )
    )

    train_dataset = Dataset(xtrain, ytrain)
    valid_dataset = Dataset(xtest, ytest)

    model = Model(
        n_inputs=data.shape[1], hidden_units_1=300, hidden_units_2=100, n_outputs=1
    )
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    schduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.78, gamma=0.3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(Config.EPOCHS):
        Engine.train(
            epoch=epoch,
            model=model,
            dataset=train_dataset,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=schduler,
        )
        Engine.eval(epoch=epoch, model=model, dataset=valid_dataset)
        torch.save(model, "model.bin")

    predictions = Engine.generate_predictions(model=model, dataset=valid_dataset)
    confus_mat = metrics.confusion_matrix(ytest, predictions)
    sns.heatmap(confus_mat, annot=True)
    plt.show()
