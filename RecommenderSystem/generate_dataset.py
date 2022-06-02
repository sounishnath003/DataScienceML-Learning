"""
# _* coding: utf8 *_

filename: generate_dataset.py

@author: sounishnath
createdAt: 2022-06-02 23:03:03
"""

from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from sklearn import model_selection

if __name__ == "__main__":
    args = Namespace(size=30_000, seed=0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    user_ids = np.random.randint(low=100, high=200, size=(args.size))
    movie_ids = np.random.randint(low=1000, high=1400, size=(args.size))
    ratings = np.round(np.random.uniform(low=1, high=5, size=(args.size)), 1)

    df = pd.DataFrame({"user": user_ids, "movie": movie_ids, "rating": ratings})
    df["kfold"] = -1
    df = df.sample(frac=1)
    X = df.drop(["rating"], axis=1)
    y = df["rating"]
    skf = model_selection.StratifiedShuffleSplit(n_splits=5)
    for fold, (trn_indx, valid_indx) in enumerate(skf.split(X, y)):
        df.loc[trn_indx, "kfold"] = fold

    df.to_csv("./inputs/dataset.csv", index=False)
