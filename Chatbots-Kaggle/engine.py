"""
# _* coding: utf8 *_

filename: engine.py

@author: sounishnath
createdAt: 2022-12-13 22:15:05
"""


import torch.nn.functional as F
from sklearn import metrics

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Engine:
    @staticmethod
    def train(model, dataset, criterion, optimizer, train_bs, epoch):
        model.to(DEVICE)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_bs,
        )
        losses = 0
        scores = {}
        for batch_idx, tdata in enumerate(dataloader):
            data = tdata["token_type_ids"].to(DEVICE)
            target = tdata["tag"].to(DEVICE)
            optimizer.zero_grad()
            yout = model(data, target)
            loss_fn = criterion(yout, target)
            losses += loss_fn.item()
            scores = Engine.compute_metrics(yout, target)
            loss_fn.backward()
            optimizer.step()

        print(
            "[TRAIN]: epoch : {0} , batch: [{1}/{2}] , loss : {3} , score : {4}".format(
                epoch,
                batch_idx + 1,
                len(dataloader),
                (losses / len(dataloader)),
                scores,
            )
        )

    @staticmethod
    def eval(model, dataset, train_bs):
        model.to(DEVICE)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_bs,
        )
        scores = {}
        predictions = []

        model.eval()
        with torch.no_grad():
            for batch_idx, tdata in enumerate(dataloader):
                data = tdata["token_type_ids"].to(DEVICE)
                target = tdata["tag"].to(DEVICE)
                ypreds_raw = model(data, target)
                scores = Engine.compute_metrics(ypreds_raw, target)
                ypreds = (
                    torch.softmax(ypreds_raw, dim=1)
                    .argmax(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                predictions.append(ypreds)
                print(
                    "[VALID]: batch: {0}/{1} ; scores: {2}".format(
                        batch_idx, len(dataloader), scores
                    )
                )

            return predictions

    @staticmethod
    def compute_metrics(yout, target=None):
        if target is None:
            return {}
        yout = yout.detach().cpu().numpy().argmax(axis=1)
        target = target.detach().cpu().numpy()

        return {
            "accuracy_score": metrics.accuracy_score(target, yout),
            "f1": metrics.f1_score(target, yout, average="macro"),
            "precision": metrics.precision_score(target, yout, average="macro"),
            "recall": metrics.recall_score(target, yout, average="macro"),
        }
