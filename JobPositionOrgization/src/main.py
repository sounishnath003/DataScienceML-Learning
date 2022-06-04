"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2022-06-04 01:04:48
"""

# Kaggle: https://www.kaggle.com/datasets/iamsouravbanerjee/software-professional-salaries-2022

import pandas as pd
import tez
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics, model_selection, preprocessing


def get_ksplit_dataframe(df, cols, fold=0):
    # df_train = df[df.kfold != fold][cols]
    # df_valid = df[df.kfold == fold][cols]
    # return df_train, df_valid
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.10, random_state=0, stratify=df["Company Name"]
    )
    df_train = df_train[cols]
    df_valid = df_valid[cols]
    return df_train, df_valid


class Dataset:
    def __init__(self, ratings, company_names, job_titles, salaries, locations) -> None:
        self.ratings = ratings
        self.company_names = company_names
        self.job_titles = job_titles
        self.salaries = salaries
        self.locations = locations

    def __len__(self):
        return len(self.company_names)

    def __getitem__(self, index):
        rating = self.ratings[index]
        company_name = self.company_names[index]
        job_title = self.job_titles[index]
        salary = self.salaries[index]
        location = self.locations[index]

        return dict(
            rating=torch.tensor(rating, dtype=torch.float),
            job=torch.tensor(job_title, dtype=torch.long),
            salary=torch.tensor(salary, dtype=torch.float),
            location=torch.tensor(location, dtype=torch.long),
            company=torch.tensor(company_name, dtype=torch.long),
        )


class CompanyRecomenderModel(tez.Model):
    def __init__(self, n_job_title, n_location, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cat: ['Job Title', 'Location']
        self.jobs_embeddings = nn.Embedding(n_job_title+1, 32)
        self.location_embeddings = nn.Embedding(n_location+1, 32)
        self.normalizer = nn.BatchNorm1d(66)
        self.dense1 = nn.Linear(66, 128)
        self.dropout = nn.Dropout(p=0.10)
        self.classifer = nn.Linear(128, n_classes)
        self.step_scheduler_after = "epoch"
        self.best_accuracy=0

    def fetch_optimizer(self, *args, **kwargs):
        opt = torch.optim.Adam(params=self.parameters(), lr=1e-3)
        return opt

    def fetch_scheduler(self, *args, **kwargs):
        sch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.72)
        return sch

    def monitor_metrics(self, y_out, original, *args, **kwargs):
        original = original.detach().cpu().numpy()
        y_out = y_out.detach().cpu().numpy().argmax(axis=1)
        accuracy = metrics.accuracy_score(y_true=original, y_pred=y_out)
        f1_macro = metrics.f1_score(y_true=original, y_pred=y_out, average="macro")
        return dict(accuracy=accuracy, f1_macro=f1_macro)

    def forward(self, rating, job, salary, location, company, *args, **kwargs):
        job_embds = self.jobs_embeddings(job)
        # print(job_embds.size())
        loc_embds = self.location_embeddings(location)
        # print(loc_embds.size())
        concated = torch.concat(
            [job_embds, loc_embds, rating.view(-1, 1), salary.view(-1, 1)], dim=1
        )
        # print(concated.size())
        normalized = self.normalizer(concated)
        # print(normalized.size())
        x = F.relu6(self.dense1(normalized))
        # print(x.size())
        x = self.dropout(x)
        # print(x.size())
        y_out = self.classifer(x)

        compute_loss = nn.CrossEntropyLoss()(y_out, company)
        compute_metrics = self.monitor_metrics(y_out, company)

        with torch.no_grad():
            if compute_metrics.get('accuracy') > self.best_accuracy:
                self.save("model.bin", weights_only=True)
                self.best_accuracy=compute_metrics.get('accuracy')
                print(f'new model weights saved with score={self.best_accuracy:0.2f} at epoch: {self.current_epoch}')
        return y_out, compute_loss, compute_metrics


if __name__ == "__main__":
    df = pd.read_csv("../input/train-folds.csv")
    column_names = ["Rating", "Company Name", "Job Title", "Salary", "Location"]
    categorical_columns = [col for col in column_names if df[col].dtype == "object"]
    print("categorical_columns=", categorical_columns)

    for cat_col in categorical_columns:
        lbl_enc = preprocessing.LabelEncoder()
        df[cat_col] = lbl_enc.fit_transform(df[cat_col].values)

    mnmx_sclr = preprocessing.MinMaxScaler()
    df.loc[:, "Salary"] = mnmx_sclr.fit_transform(df["Salary"].values.reshape(-1, 1))

    mnmx_sclr = preprocessing.MinMaxScaler()
    df.loc[:, "Rating"] = mnmx_sclr.fit_transform(df["Rating"].values.reshape(-1, 1))

    df_train, df_valid = get_ksplit_dataframe(df, column_names, fold=0)
    train_dataset = Dataset(
        ratings=df_train["Rating"].values,
        company_names=df_train["Company Name"].values,
        job_titles=df_train["Job Title"].values,
        salaries=df_train["Salary"].values,
        locations=df_train["Location"].values,
    )
    valid_dataset = Dataset(
        ratings=df_valid["Rating"].values,
        company_names=df_valid["Company Name"].values,
        job_titles=df_valid["Job Title"].values,
        salaries=df_valid["Salary"].values,
        locations=df_valid["Location"].values,
    )

    print(
        {
            "train_shape": df_train.shape,
            "valid_shape": df_valid.shape,
        }
    )

    model = CompanyRecomenderModel(
        n_job_title=df["Job Title"].nunique(),
        n_location=df["Location"].nunique(),
        n_classes=df["Company Name"].nunique(),
    )
    print(model)

    model.fit(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_bs=128,
        valid_bs=64,
        device="mps",
        epochs=50,
    )
    model.save("model.bin")
