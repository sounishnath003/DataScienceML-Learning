

import pandas as pd
import torch
import torch.utils
from sklearn import preprocessing
from torch import nn

feature_scaler = preprocessing.MinMaxScaler(clip=True)
target_scaler = preprocessing.MinMaxScaler(clip=True)


# "data/california_housing_train.csv"
# "median_house_value"
def read_dataset(file, target_col, transform=True):
    X = pd.read_csv(file)
    target = torch.reshape(torch.tensor(X[target_col]), shape=(-1,1))

    if transform:
        X = feature_scaler.fit_transform(X.iloc[:, :-1])
        target = target_scaler.fit_transform(target)
    else:
        X = feature_scaler.transform(X.iloc[:, :-1])
        target = target_scaler.transform(target)

    return dict(
        X=X,
        target=target,
    )


train_dataset = read_dataset(
    file="data/california_housing_train.csv",
    target_col="median_house_value"
)

test_dataset = read_dataset(
    file="data/california_housing_test.csv",
    target_col="median_house_value",
    transform=False
)


class Dataset:
    def __init__(self, data, targets) -> None:
        super().__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]
    
    def shape(self):
        return self.data.shape

    def __getitem__(self, index):
        data = self.data[index, :]
        target = self.targets[index]

        return dict(
            data=torch.tensor(data, dtype=torch.float, device="cuda"),
            target=torch.tensor(target, dtype=torch.float, device="cuda")
        )


train_dataset = Dataset(
    data=train_dataset["X"], targets=train_dataset["target"])
test_dataset = Dataset(data=test_dataset["X"], targets=test_dataset["target"])


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32)

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(8, 8)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(8, 100)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(100, 32)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(32, 1)

    def forward(self, X):
        x = self.flatten(X)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        logits = self.output(x)
        return logits


def train_loop(dataloader, model, optimizer, loss_fn, epoch):
    size = len(dataloader.dataset)
    epoch_loss = 0.0
    for data in dataloader:
        X = data["data"]
        y = data["target"]
        yhat = model(X)
        loss = loss_fn(yhat, y) # torch.mean(y.view(-1) - yhat.view(-1)).pow(2)
        epoch_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if epoch % 50 == 0:
    print (f"epoch {epoch}, loss: {epoch_loss/epoch}")


model = Model().to("cuda")
print (model)


loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 1000

for t in range(epochs):
    train_loop(dataloader=train_loader, model=model, optimizer=optimizer, loss_fn=loss_fn, epoch=t+1)
print("Done!")



"""
Model(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear1): Linear(in_features=8, out_features=8, bias=True)
  (relu1): ReLU()
  (linear2): Linear(in_features=8, out_features=100, bias=True)
  (relu2): ReLU()
  (linear3): Linear(in_features=100, out_features=32, bias=True)
  (relu3): ReLU()
  (output): Linear(in_features=32, out_features=1, bias=True)
)
epoch 1, loss: 43.1380615234375
epoch 2, loss: 15.236654281616211
epoch 3, loss: 10.137258529663086
epoch 4, loss: 7.598879337310791
epoch 5, loss: 6.071691513061523
epoch 6, loss: 5.054876327514648
epoch 7, loss: 4.327295780181885
epoch 8, loss: 3.7841217517852783
epoch 9, loss: 3.361783504486084
epoch 10, loss: 3.023909091949463
epoch 11, loss: 2.7474708557128906
epoch 12, loss: 2.5170936584472656
epoch 13, loss: 2.322164297103882
epoch 14, loss: 2.155057430267334
epoch 15, loss: 2.010221004486084
epoch 16, loss: 1.8834935426712036
epoch 17, loss: 1.7716516256332397
epoch 18, loss: 1.6722162961959839
epoch 19, loss: 1.5832051038742065
epoch 20, loss: 1.5031194686889648
epoch 21, loss: 1.4306249618530273
epoch 22, loss: 1.3647271394729614
epoch 23, loss: 1.304520606994629
epoch 24, loss: 1.2493047714233398
epoch 25, loss: 1.1984883546829224
epoch 26, loss: 1.1515629291534424
epoch 27, loss: 1.1080752611160278
epoch 28, loss: 1.0676813125610352
epoch 29, loss: 1.0300450325012207
epoch 30, loss: 0.9948917031288147
epoch 31, loss: 0.9619755148887634
epoch 32, loss: 0.9310941696166992
epoch 33, loss: 0.9020535945892334
epoch 34, loss: 0.8746879696846008
epoch 35, loss: 0.8488581776618958
epoch 36, loss: 0.8244350552558899
epoch 37, loss: 0.8012993335723877
epoch 38, loss: 0.77934730052948
epoch 39, loss: 0.7584884166717529
epoch 40, loss: 0.7386342287063599
epoch 41, loss: 0.7197131514549255
epoch 42, loss: 0.7016538977622986
epoch 43, loss: 0.6843894720077515
epoch 44, loss: 0.667856752872467
epoch 45, loss: 0.6520127058029175
epoch 46, loss: 0.636812686920166
epoch 47, loss: 0.6222158074378967
epoch 48, loss: 0.6081916093826294
epoch 49, loss: 0.59470134973526
epoch 50, loss: 0.5817198157310486

"""