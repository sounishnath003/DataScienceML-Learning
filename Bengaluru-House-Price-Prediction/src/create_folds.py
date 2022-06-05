"""
# _* coding: utf8 *_

filename: create_folds.py

@author: sounishnath
createdAt: 2022-06-05 15:08:18
"""

import re

import pandas as pd
from sklearn import model_selection


def change_availability(avalability):
    if avalability != "Ready To Move":
        return "Requires-Time"
    return "Ready To Move"


def convert_to_number(val):
    try:
        return float(val)
    except Exception:
        val = re.findall("\d+", val.split(" ")[0])
        if len(val) > 1:
            return float(val[0])
        else:
            return float(val[0])


if __name__ == "__main__":
    df = pd.read_csv("../input/dataset.csv")
    df.loc[:, "availability"] = df.availability.apply(change_availability)
    df.loc[:, "total_sqft"] = df.total_sqft.apply(convert_to_number)
    df = df.fillna(method="bfill")
    df["kfold"] = -1
    df = df.sample(frac=1.0)
    X = df.drop(["size"], axis=1)
    y = df["size"]
    skf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (trn_, valid_) in enumerate(skf.split(X, y)):
        df.loc[valid_, "kfold"] = fold
    df.to_csv("../input/train-folds.csv", index=False)
    print(df.sample(10))
