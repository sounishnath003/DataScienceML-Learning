"""
# _* coding: utf8 *_

filename: movielence_dataset.py

@author: sounishnath
createdAt: 2023-10-28 20:33:24
"""

import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler, random_split
from transformers import AutoTokenizer


class MovieLenseDataset(Dataset):
    def __init__(
        self,
        dataset_directory: str,
        tokenizer_name: str,
        max_length: int = 32,
        n_rows: int = None,
    ) -> None:
        super(MovieLenseDataset, self).__init__()
        self.dataset_folder = dataset_directory
        self.max_length = max_length
        self.info_dataset_path = os.path.join(dataset_directory, "u.info")
        self.genre_dataset_path = os.path.join(dataset_directory, "u.genre")
        self.occupation_dataset_path = os.path.join(dataset_directory, "u.occupation")
        self.data_dataset_path = os.path.join(dataset_directory, "u.data")
        self.item_dataset_path = os.path.join(dataset_directory, "u.item")
        self.user_dataset_path = os.path.join(dataset_directory, "u.user")

        self.info_dfx = pd.read_csv(self.info_dataset_path, sep=" ", header=None)
        self.info_dfx.columns = ["Counts", "Type"]

        self.genre_dfx = pd.read_csv(
            self.genre_dataset_path, sep="|", encoding="latin-1", header=None
        )
        self.genre_dfx.drop(self.genre_dfx.columns[1], axis=1, inplace=True)

        self.occupation_dfx = pd.read_csv(
            self.occupation_dataset_path, sep="|", encoding="latin-1", header=None
        )
        self.occupation_dfx.columns = ["Occupations"]

        self.data_dfx = pd.read_csv(self.data_dataset_path, sep="\t", header=None)
        self.data_dfx.columns = ["user id", "movie id", "rating", "timestamp"]

        self.item_dfx = pd.read_csv(
            self.item_dataset_path, sep="|", encoding="latin-1", header=None
        )
        self.item_dfx.columns = [
            "movie id",
            "movie title",
            "release date",
            "video release date",
            "IMDb URL",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]

        self.user_dfx = pd.read_csv(
            self.user_dataset_path, sep="|", encoding="latin-1", header=None
        )
        self.user_dfx.columns = ["user id", "age", "gender", "occupation", "zip code"]

        # merge 3 dfx start
        # merge the 'data' table with 'user' table
        data_user = pd.merge(
            self.data_dfx,
            self.user_dfx,
            on="user id",
        )
        data_user.drop(columns=["user id"], inplace=True)

        # merge the 'Data_User' dataframe with 'Item' dataframe to get each rating, occupation of user and movie title
        self.data_user_item_dfx = pd.merge(
            data_user,
            self.item_dfx,
            on="movie id",
        )
        self.data_user_item_dfx.drop(columns=["movie id"], inplace=True)
        if not n_rows is None:
            self.data_user_item_dfx = self.data_user_item_dfx.iloc[:n_rows, :].sample(
                frac=1
            )
        # merge 3 dfx end

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data_user_item_dfx)

    def __getitem__(self, index):
        data = self.data_user_item_dfx.loc[index, :]
        genres = []
        for col in [
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]:
            if data[col] == 1:
                genres.append(col)

            user_detail = self.tokenizer.encode_plus(
                "My age is {0} . My gender is {1} . My occupation is {2}".format(
                    data["age"],
                    "Male" if data["gender"] == "M" else "Female",
                    data["occupation"],
                ),
                None,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True,
            )
            user_detail = {
                "input_ids": torch.tensor(user_detail["input_ids"], dtype=torch.long),
                "token_type_ids": torch.tensor(
                    user_detail["token_type_ids"], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    user_detail["attention_mask"], dtype=torch.long
                ),
            }

            content_detail = self.tokenizer.encode_plus(
                "The movie title is {0} . The genre of the movie is {1} . I have given the movie-rating to {2}".format(
                    data["movie title"], ", ".join(genres), (data["rating"] - 1)
                ),
                None,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True,
            )
            content_detail = {
                "input_ids": torch.tensor(
                    content_detail["input_ids"], dtype=torch.long
                ),
                "token_type_ids": torch.tensor(
                    content_detail["token_type_ids"], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    content_detail["attention_mask"], dtype=torch.long
                ),
            }

        return dict(
            rating=torch.tensor(data["rating"] - 1, dtype=torch.long),
            user_detail=user_detail,
            content_detail=content_detail,
        )

    def train_test_split(self, dataset, val_size: float = 0.20):
        train_size = int((1.0 - val_size) * len(dataset))
        val_size = int((val_size) * len(self))

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return (train_dataset, val_dataset)
