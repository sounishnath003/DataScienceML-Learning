"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2022-06-06 18:14:51
"""

import re

import pandas as pd
import tez
from sklearn import model_selection, preprocessing

import config
from dataset import Dataset
from model import TextClassifierModel

# Kaggle: https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


def get_folded_data(preprocess_text):
    dfx = pd.read_csv(config.TRAINING_DATASET)[["text", "sentiment"]]
    dfx.text = dfx.text.astype(str)
    dfx.sentiment = dfx.sentiment.astype(str)
    dfx.text = dfx.text.apply(preprocess_text)
    lbl_enc = preprocessing.LabelEncoder()
    dfx.sentiment = lbl_enc.fit_transform(dfx.sentiment.values)
    dfx["kfold"] = -1
    dfx = dfx.sample(frac=1).reset_index(drop=True)
    skf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (trn_, vld_) in enumerate(
        skf.split(X=dfx.text.values, y=dfx.sentiment.values)
    ):
        dfx.loc[vld_, "kfold"] = fold
    fold = 1
    df_train = dfx[dfx.kfold != fold].drop(["kfold"], axis=1).reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].drop(["kfold"], axis=1).reset_index(drop=True)
    print(
        dict(
            train_size=df_train.shape,
            valid_size=df_valid.shape,
        )
    )

    return df_train, df_valid, len(lbl_enc.classes_)


if __name__ == "__main__":
    df_train, df_valid, n_classes = get_folded_data(preprocess_text)
    train_dataset = Dataset(
        text=df_train.text.values, sentiment=df_train.sentiment.values
    )
    valid_dataset = Dataset(
        text=df_valid.text.values, sentiment=df_valid.sentiment.values
    )
    model = TextClassifierModel(
        model_name=config.BERT_PATH,
        num_train_steps=(
            len(train_dataset)
            / config.TRAIN_BATCH_SIZE
            / (config.ACCUMULATION_STEP * config.EPOCHS)
        ),
        learning_rate=config.LEARNING_RATE,
        num_classes=n_classes,
    )
    model = tez.Tez(model)
    model_configuration = tez.TezConfig(
        device=config.DEVICE,
        training_batch_size=config.TRAIN_BATCH_SIZE,
        validation_batch_size=config.VALID_BATCH_SIZE,
        epochs=config.EPOCHS,
        gradient_accumulation_steps=config.ACCUMULATION_STEP,
        clip_grad_norm=1.0,
        step_scheduler_after="batch",
    )
    es = tez.callbacks.EarlyStopping(
        monitor="valid_loss",
        model_path=config.MODEL_PATH,
        patience=3,
        save_weights_only=True,
    )
    model.fit(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        config=model_configuration,
        callbacks=[es],
    )
