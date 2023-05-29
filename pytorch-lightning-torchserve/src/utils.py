"""
# _* coding: utf8 *_

filename: utils.py

@author: sounishnath
createdAt: 2023-05-22 23:46:27
"""

import json

from sklearn import model_selection


def to_string(anydata) -> str:
    return json.dumps({"data": str(anydata)})


def split_dataset(dfx, n_splits: int = 5, fold: int = 0):
    dfx.loc[:, "kfold"] = -1
    skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
    for batch, (trn_index, valid_index) in enumerate(
        skf.split(dfx["review"].values, y=dfx["sentiment"].values)
    ):
        dfx.loc[valid_index, "kfold"] = batch
    return (
        dfx[dfx["kfold"] != fold][["review", "sentiment"]]
        .copy()
        .reset_index(drop=True),
        dfx[dfx["kfold"] == fold][["review", "sentiment"]]
        .copy()
        .reset_index(drop=True),
    )
