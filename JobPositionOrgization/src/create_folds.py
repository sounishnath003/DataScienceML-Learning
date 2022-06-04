"""
# _* coding: utf8 *_

filename: create_folds.py

@author: sounishnath
createdAt: 2022-06-04 01:06:50
"""

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/dataset.csv")
    print("size=", df.shape)
    column_names = ["Rating", "Company Name", "Job Title", "Salary", "Location"]
    df = df[column_names]
    company_names = set(
        df["Company Name"]
        .value_counts()[df["Company Name"].value_counts() > 9]
        .keys()
        .tolist()   
    )
    df = df[df["Company Name"].isin(company_names)].reset_index(drop=True)
    df.sample(frac=1.0)
    X = df.drop(["Company Name"], axis=1)
    y = df["Company Name"]

    df["kfold"] = -1
    skf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (trn_, vld_) in enumerate(skf.split(X, y)):
        df.loc[trn_, "kfold"] = fold

    df.to_csv("../input/train-folds.csv", index=False)
    print("size=", df.shape)
