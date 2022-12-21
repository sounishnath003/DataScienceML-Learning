"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2022-12-21 09:12:56
"""

import pandas as pd
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing, model_selection, metrics
from engine import Engine


def create_fold(dfx: pd.DataFrame, target_column: str, k=5):
    dfx = dfx.dropna(axis=0).reset_index(drop=True)

    dfx["fold"] = -1
    dfx = dfx.sample(frac=1.0).reset_index(drop=True)
    X = dfx.drop([target_column], axis=1)
    y = dfx[target_column]

    skf = model_selection.StratifiedKFold(n_splits=k, shuffle=True, random_state=31)
    for fold, (trn_, valid_) in enumerate(skf.split(X=X, y=y)):
        dfx.loc[valid_, "fold"] = fold

    dfx.query("fold!=0").reset_index(drop=True).to_csv("train_data.csv", index=False)
    dfx.query("fold==0").reset_index(drop=True).to_csv("valid_data.csv", index=False)

    tdfx = pd.read_csv("train_data.csv")
    train_data = tdfx.query("fold!=0").drop([target_column, "fold"], axis=1).values
    train_targets = tdfx.query("fold!=0")[target_column].values

    tdfx = pd.read_csv("valid_data.csv")
    valid_data = tdfx.query("fold==0").drop([target_column, "fold"], axis=1).values
    valid_targets = tdfx.query("fold==0")[target_column].values

    return {
        "train_data": train_data,
        "train_targets": train_targets,
        "valid_data": valid_data,
        "valid_targets": valid_targets,
    }


class Dataset:
    def __init__(self, data, target) -> None:
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        return {
            "data": torch.tensor(self.data[item], dtype=torch.float),
            "target": torch.tensor(self.target[item], dtype=torch.float),
        }


class Model(nn.Module):
    def __init__(self, n_inputs) -> None:
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(512, 1),
        )

    def forward(self, data, target=None):
        out = self.model(data)
        return out


def run():
    dfx = pd.read_csv("./src/adult.csv")
    TARGET_COLUMN = "income"

    numeric_columns = [
        (col, preprocessing.StandardScaler())
        for col in dfx.columns
        if dfx[col].dtype == "int64" and col != TARGET_COLUMN
    ]
    categorical_columns = [
        (col, preprocessing.LabelEncoder())
        for col in dfx.columns
        if dfx[col].dtype == "object" and col != TARGET_COLUMN
    ]

    for col, sclr in numeric_columns:
        dfx.loc[:, col] = sclr.fit_transform(dfx[[col]])

    for col, lbl_enc in categorical_columns:
        dfx.loc[:, col] = lbl_enc.fit_transform(dfx[[col]])

    target_encoding = {"<=50K": 0, ">50K": 1}
    inverse_target_encoding = {"<=50K": 0, ">50K": 1}
    dfx.loc[:, TARGET_COLUMN] = dfx[TARGET_COLUMN].map(target_encoding)

    data = create_fold(dfx, target_column=TARGET_COLUMN, k=3)
    train_dataset = Dataset(data=data["train_data"], target=data["train_targets"])
    valid_dataset = Dataset(data=data["valid_data"], target=data["valid_targets"])

    print(
        dict(
            train_size=len(train_dataset),
            valid_size=len(valid_dataset),
            shape=train_dataset[0]["data"].size(),
        )
    )

    model = Model(n_inputs=train_dataset[0]["data"].size(0))
    EPOCHS = 10
    optim = torch.optim.SGD(params=model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    try:
        model.load_state_dict(torch.load("model.pth"))
        print("model loaded from weights")
        Engine.eval(model=model, dataset=valid_dataset, train_bs=32)
    except Exception as e:
        for epoch in range(EPOCHS):
            Engine.train(
                model=model,
                dataset=train_dataset,
                criterion=criterion,
                optimizer=optim,
                train_bs=32,
                epoch=epoch,
            )
            Engine.eval(model=model, dataset=valid_dataset, train_bs=32)

            torch.save(model.state_dict(), "model.pth")
            print(f"model.pth saved at epoch {epoch}....")

        torch.save(model.state_dict(), "model.pth")
        print(f"model.pth saved finally....")

    predictions = Engine.eval(model=model, dataset=valid_dataset, train_bs=32)
    cm = metrics.confusion_matrix(data["valid_targets"], predictions)
    print(cm)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run()
