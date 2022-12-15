"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2022-12-13 09:22:02
"""

import json
import random
import typing
import warnings

import pandas as pd
import torch.nn as nn
from sklearn import model_selection, preprocessing

import torch
from dataset import IntentDataset
from engine import Engine
from intent import IntentModel
from model import IntentDLModel, IntentRnnModel
from tokenizer import Tokenizer


def load_intent_file() -> typing.List[IntentModel]:
    with open("intents.json", "r") as file:
        intent_json_payload = json.load(file)
        file.close()

    intent_data = []
    for payload in intent_json_payload["intents"]:
        intent = IntentModel(
            intent=payload["intent"],
            text=payload["text"],
            responses=payload["responses"],
        )
        intent_data.append(intent)

    return intent_data


def create_dataset(intent_data: typing.List[IntentModel]) -> pd.DataFrame:
    tags = []
    intents = []

    for intent in intent_data:
        wx = [sent for sent in intent.text]
        intents.extend(wx)
        for _ in range(len(wx)):
            tags.append(intent.intent)

    dfx = (
        pd.DataFrame({"intents": intents, "tags": tags})
        .sample(frac=1)
        .reset_index(drop=True)
    )
    print("dataset_shape=", dfx.shape)
    return dfx


def perform_prediction(
    model: IntentRnnModel,
    tokenizer: Tokenizer,
    sentence: str,
    lbl_encoder: preprocessing.LabelEncoder,
    intent_data: typing.List[IntentModel],
):
    tokenized_dict = tokenizer.tokenize(sentence)
    token_type_ids = torch.tensor(
        tokenized_dict["token_type_ids"], dtype=torch.long
    ).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(token_type_ids)
        output = torch.softmax(output, dim=1)
        probs = output.max(dim=1)[0]
        output = output.argmax(dim=1)
        intent = lbl_encoder.inverse_transform(output.cpu().numpy())[0]
        roboresponse = [
            random.choice(intt.responses)
            for intt in intent_data
            if intt.intent == intent
        ][0]
        print(
            "[BOT]:",
            {
                "sentence": sentence,
                "predicted_intent": intent,
                "probability": round(probs.item(), 3),
                "bot_response": roboresponse,
            },
        )


def run():
    # nltk.download("punkt")
    # nltk.download("wordnet")

    intent_datas = load_intent_file()
    dfx = create_dataset(intent_datas)
    lbl_encoder = preprocessing.LabelEncoder()
    dfx.loc[:, "encoded_tags"] = lbl_encoder.fit_transform(dfx.tags.values)
    dfx["fold"] = -1

    skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (trn_, valid_) in enumerate(
        skf.split(dfx.drop("encoded_tags", axis=1), dfx.encoded_tags.values)
    ):
        dfx.loc[valid_, "fold"] = fold
    train_dfx, valid_dfx = dfx.query("fold!=0").reset_index(drop=True), dfx.query(
        "fold==0"
    ).reset_index(drop=True)
    print(dict(train=train_dfx.shape, valid=valid_dfx.shape))

    tokenizer = Tokenizer(texts=dfx.intents.values, pad_length=18)
    print(tokenizer.tokenize("it was nice talking to you good talk"))
    train_dataset = IntentDataset(
        intents=train_dfx.intents.values,
        tags=train_dfx.encoded_tags.values,
        tokenizer=tokenizer,
    )
    valid_dataset = IntentDataset(
        intents=valid_dfx.intents.values,
        tags=valid_dfx.encoded_tags.values,
        tokenizer=tokenizer,
    )

    # model = IntentDLModel(
    #     n_embeddings=len(tokenizer.vocabset),
    #     n_embedding_dim=512,
    #     padding_idx=57,
    #     n_hidden_layer=3,
    #     n_hidden_layer_neurons=512,
    #     n_classes=dfx.encoded_tags.nunique(),
    # )
    model = IntentRnnModel(
        n_embeddings=len(tokenizer.vocabset) + 1,
        n_embedding_dim=512,
        padding_idx=123,
        n_hidden_layer=3,
        n_hidden_layer_neurons=512,
        n_classes=dfx.encoded_tags.nunique(),
    )
    print(model)

    EPOCHS = 10
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        Engine.train(
            model=model,
            dataset=train_dataset,
            criterion=criterion,
            optimizer=optim,
            train_bs=8,
            epoch=epoch,
        )
        Engine.eval(model=model, dataset=valid_dataset, train_bs=16)
        perform_prediction(
            model=model,
            tokenizer=tokenizer,
            sentence="it was nice talking to you good talk",
            lbl_encoder=lbl_encoder,
            intent_data=intent_datas,
        )

    inp = input("Enter your sentence: ")
    while inp != "Q":
        perform_prediction(
            model=model,
            tokenizer=tokenizer,
            sentence=inp,
            lbl_encoder=lbl_encoder,
            intent_data=intent_datas,
        )
        inp = input("[YOU]: ")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run()
