"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2022-06-05 14:02:25
"""

# Kaggle: https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data

import numpy as np
import pandas as pd
import tez
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics, preprocessing


def get_folded_df(df, fold=0):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    df_train.drop(["kfold"], axis=1, inplace=True)
    df_valid.drop(["kfold"], axis=1, inplace=True)
    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True)


class Dataset:
    def __init__(
        self,
        area_types,
        availabilitys,
        locations,
        sizes,
        societies,
        total_sqfts,
        baths,
        balconys,
        prices=None,
    ) -> None:
        self.area_types = area_types
        self.availabilitys = availabilitys
        self.locations = locations
        self.sizes = sizes
        self.societies = societies
        self.total_sqfts = total_sqfts
        self.baths = baths
        self.balconys = balconys
        self.prices = prices

    def __len__(self):
        return len(self.area_types)

    def __getitem__(self, index):
        area_type = self.area_types[index]
        availability = self.availabilitys[index]
        location = self.locations[index]
        size = self.sizes[index]
        society = self.societies[index]
        total_sqft = self.total_sqfts[index]
        bath = self.baths[index]
        balcony = self.balconys[index]
        if self.prices is not None:
            price = self.prices[index]
        else:
            price = 0

        return {
            "area_type": torch.tensor(area_type, dtype=torch.long),
            "availability": torch.tensor(availability, dtype=torch.long),
            "location": torch.tensor(location, dtype=torch.long),
            "size": torch.tensor(size, dtype=torch.long),
            "society": torch.tensor(society, dtype=torch.long),
            "total_sqft": torch.tensor(total_sqft, dtype=torch.float),
            "bath": torch.tensor(bath, dtype=torch.float),
            "balcony": torch.tensor(balcony, dtype=torch.float),
            "price": torch.tensor(price, dtype=torch.float),
        }


class RoomPriceRegressor(tez.Model):
    def __init__(
        self,
        n_areas,
        n_availabilities,
        n_locations,
        n_sizes,
        n_societies,
        *args,
        **kwargs,
    ):
        super(RoomPriceRegressor, self).__init__(*args, **kwargs)
        self.area_embedding_layer = nn.Embedding(
            n_areas + 1, int(max(n_areas // 2, 64))
        )
        self.availability_embedding_layer = nn.Embedding(
            n_availabilities + 1, int(max(n_availabilities // 2, 64))
        )
        self.location_embedding_layer = nn.Embedding(
            n_locations + 1, int(min(n_locations // 2, 64))
        )
        self.size_embedding_layer = nn.Embedding(
            n_sizes + 1, int(max(n_sizes // 2, 64))
        )
        self.society_embedding_layer = nn.Embedding(
            n_societies + 1, int(min(n_societies // 2, 64))
        )
        self.normalizer_layer = nn.LayerNorm(323)
        self.batch_normalizer_layer = nn.BatchNorm1d(323)
        self.fc1 = nn.Linear(323, 512)
        self.fc2 = nn.Linear(512, 256)
        self.regressor = nn.Linear(256, 1)
        self.best_rmse = np.inf

    def fetch_scheduler(self, *args, **kwargs):
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=3, gamma=0.70
        )
        return sch

    def fetch_optimizer(self, *args, **kwargs):
        opt = torch.optim.Adam(params=self.parameters(), lr=2e-2)
        return opt

    def monitor_metrics(self, y_out, price, *args, **kwargs):
        y_out = y_out.cpu().detach().numpy()
        price = price.cpu().detach().numpy()
        rmse = np.sqrt(metrics.mean_squared_error(y_true=price, y_pred=y_out))
        return {"rmse": rmse}

    def forward(
        self,
        area_type,
        availability,
        location,
        size,
        society,
        total_sqft,
        bath,
        balcony,
        price=None,
    ):
        area_embeds = self.area_embedding_layer(area_type)
        availability_embeds = self.availability_embedding_layer(availability)
        location_embeds = self.location_embedding_layer(location)
        size_embeds = self.size_embedding_layer(size)
        society_embeds = self.society_embedding_layer(society)
        concated = torch.concat(
            [
                area_embeds,
                availability_embeds,
                location_embeds,
                size_embeds,
                society_embeds,
                total_sqft.view(-1, 1),
                bath.view(-1, 1),
                balcony.view(-1, 1),
            ],
            dim=1,
        )
        x = self.normalizer_layer(concated)
        x = self.batch_normalizer_layer(x)
        x = F.relu6(self.fc1(x))
        x = F.relu6(self.fc2(x))
        y_out = F.relu6(self.regressor(x))

        compute_loss = torch.sqrt(nn.MSELoss()(y_out, price.view(-1, 1)))
        compute_metrics = self.monitor_metrics(y_out, price.view(-1, 1))

        with torch.no_grad():
            if compute_metrics.get("rmse") < self.best_rmse:
                self.save("model.bin", weights_only=True)
                print(
                    f"model.bin saved with rmse: {self.best_rmse:0.3f} at epoch: {self.current_epoch}"
                )
                self.best_rmse = compute_metrics.get("rmse")

        return y_out, compute_loss, compute_metrics


if __name__ == "__main__":
    df = pd.read_csv("../input/train-folds.csv")
    df = df.dropna(axis=0, how="any")
    loc_vc = df.location.value_counts()
    df.loc[:, "location"] = df.location.apply(
        lambda x: "OTHER" if x in loc_vc[loc_vc <= 15] else x
    )
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    print("categorical columns=", categorical_columns)
    for catcol in categorical_columns:
        lbl_enc = preprocessing.LabelEncoder()
        df[catcol] = lbl_enc.fit_transform(df[catcol])
    df_train, df_valid = get_folded_df(df, fold=0)

    train_dataset = Dataset(
        area_types=df_train.area_type.values,
        availabilitys=df_train.availability.values,
        locations=df_train.location.values,
        sizes=df_train["size"].values,
        societies=df_train.society.values,
        total_sqfts=df_train.total_sqft.values,
        baths=df_train.bath.values,
        balconys=df_train.balcony.values,
        prices=df_train.price.values,
    )
    valid_dataset = Dataset(
        area_types=df_valid.area_type.values,
        availabilitys=df_valid.availability.values,
        locations=df_valid.location.values,
        sizes=df_valid["size"].values,
        societies=df_valid.society.values,
        total_sqfts=df_valid.total_sqft.values,
        baths=df_valid.bath.values,
        balconys=df_valid.balcony.values,
        prices=df_valid.price.values,
    )
    print(
        dict(
            train_size=df_train.shape,
            valid_size=df_valid.shape,
        )
    )

    model = RoomPriceRegressor(
        n_areas=df.area_type.nunique(),
        n_availabilities=df.availability.nunique(),
        n_locations=df.location.nunique(),
        n_sizes=df["size"].nunique(),
        n_societies=df.society.nunique(),
    )
    print(model)

    es = tez.callbacks.EarlyStopping(
        monitor="train_loss", model_path="model.bin", save_weights_only=True
    )
    model.fit(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_bs=1024,
        valid_bs=512,
        epochs=2,
        device="mps",
        callbacks=[es],
    )
    model.save("model.bin")
