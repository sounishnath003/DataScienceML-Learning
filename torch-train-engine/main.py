"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2022-07-21 20:29:18
"""

from sklearn import datasets, model_selection

import torch
import torch.nn as nn
import torch.utils.data
from config import EPOCHS, TRAIN_BS, VALID_BS
from dataset import Dataset
from engine import Engine
from logger import logger
from model import Model


def generate_datasets(n_samples):
    data, targets = datasets.make_classification(
        n_samples, n_classes=5, n_informative=4, random_state=0
    )
    return data, targets


if __name__ == "__main__":
    data, target = generate_datasets(n_samples=100000)
    traindata, validdata, traintarget, validtarget = model_selection.train_test_split(
        data, target, test_size=0.05, random_state=0, stratify=target
    )

    train_dataset = Dataset(traindata, traintarget)
    valid_dataset = Dataset(validdata, validtarget)
    logger.info(train_dataset[99])
    logger.info(valid_dataset[99])

    model = Model(n_inputs=20)
    save_model = False
    try:
        model.load_state_dict(torch.load("model.pth"))
        logger.info("model weights initialized...")
        logger.info("model.pth loaded!...")
    except:
        save_model = True
    logger.info(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.42, gamma=0.33)
    predictions = []

    if save_model:
        for epoch in range(EPOCHS):
            Engine.train(
                model,
                train_dataset,
                criterion,
                optimizer,
                train_bs=TRAIN_BS,
                epoch=epoch,
            )
            sch.step()

    predictions = Engine.eval(model, valid_dataset, train_bs=VALID_BS)

    if save_model:
        torch.save(model.state_dict(), "model.pth")
        logger.info("model.pth has been saved")

    logger.info(
        {
            "prediction": predictions[-1],
            "original": valid_dataset[-VALID_BS:]["target"].cpu().detach().numpy(),
        }
    )

    logger.info(model.state_dict().keys())
