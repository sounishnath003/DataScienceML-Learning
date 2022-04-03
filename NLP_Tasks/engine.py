

import torch
import torch.utils.data
from sklearn.metrics import accuracy_score


class Engine:
    def __init__(self) -> None:
        pass

    @classmethod
    def accuracy_calculator(self, yhat, y):
        with torch.no_grad():
            y = y.type(torch.LongTensor)
            yhat = [1 if yt >= 0.51 else 0 for yt in yhat]
            print("y: ", y)
            print("yhat: ", yhat)
            score = torch.tensor(yhat == y).sum().item()
            return score / len(yhat)

    @classmethod
    def train_loop(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, criterion, optimizer, epochs):
        model.train()
        steps = len(iter(data_loader))
        for epoch in range(epochs):
            for batch, (x, y) in enumerate(data_loader):
                y = y.type(torch.FloatTensor)
                yhat = model.forward(x)
                loss = criterion(yhat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch == 1 or epoch % 5 == 0:
                    print(
                        f"[TRAIN] epoch: {epoch+1}/{epochs}, steps: {batch+1}/{steps} loss: {loss.item():0.4f}")

    @classmethod
    def eval_loop(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, criterion, optimizer, epochs):
        model.eval()
        steps = len(iter(data_loader))
        with torch.no_grad():
            for epoch in range(epochs):
                for batch, (x, y) in enumerate(data_loader):
                    y = y.type(torch.FloatTensor)
                    yhat = model.forward(x)
                    loss = criterion(yhat, y)

                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()

                    if epoch == 1 or epoch % 5 == 0:
                        print(
                            f"[EVAL] epoch: {epoch+1}/{epochs}, steps: {batch+1}/{steps} loss: {loss.item():0.4f}")
