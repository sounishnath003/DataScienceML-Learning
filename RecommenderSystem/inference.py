"""
# _* coding: utf8 *_

filename: inference.py

@author: sounishnath
createdAt: 2022-06-02 23:02:56
"""

from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from tqdm import tqdm

from main import RecommenderSystemModel


def get_fold_dataset(df, kfold):
    return df[df.kfold == kfold][["user", "movie", "rating"]].reset_index(drop=True)

class Dataset:
    def __init__(self, users, movies, ratings) -> None:
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        movie = self.movies[index]
        rating = self.ratings[index]
        return dict(
            user=torch.tensor(user, dtype=torch.long),
            movie=torch.tensor(movie, dtype=torch.long),
            rating=torch.tensor(rating, dtype=torch.float),
        )


if __name__ == "__main__":
    args = Namespace(size=30_000, seed=0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv("./inputs/dataset.csv")
    df.user = preprocessing.LabelEncoder().fit_transform(df.user.values)
    df.movie = preprocessing.LabelEncoder().fit_transform(df.movie.values)

    df_valid=get_fold_dataset(df,1)
    valid_dataset = Dataset(
        users=df_valid.user.values,
        movies=df_valid.movie.values,
        ratings=df_valid.rating.values,
    )

    n_users = df.user.nunique()
    n_movies = df.movie.nunique()
    model=RecommenderSystemModel(n_users=n_users, n_movies=n_movies)
    model.load('model.bin', device='mps')
    
    predictions=model.predict(valid_dataset)
    tk0=tqdm(predictions)
    predics=[]
    for pred in tk0:
        predics.extend(np.round(pred.ravel(),1).tolist())
    
    df_valid['predict']=predics
    df_valid.to_csv('prediction.csv', index=False)
    