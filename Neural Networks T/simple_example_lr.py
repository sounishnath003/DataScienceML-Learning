"""
# _* coding: utf8 *_

filename: simple_example.py

@author: sounishnath
createdAt: 2022-04-15 12:25:33
"""

import numpy as np
from sklearn import datasets, linear_model
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

np.random.seed(0)
torch.random.manual_seed(0)

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = nn.Linear(288, 288 * 2)
        self.dense2 = nn.Linear(288 * 2, 288 * 2)
        self.dense3 = nn.Linear(288 * 2, 288 * 2)
        self.dense4 = nn.Linear(288 * 2, 1)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dense3(x)
        x = F.relu(x)
        outputs = self.dense4(x)
        return outputs


class Dataset:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        x = self.X[item]
        y = self.y[item]

        return dict(
            data=torch.tensor(x, dtype=torch.float),
            target=torch.tensor(y, dtype=torch.float)
        )

if __name__ == '__main__':
    X, y = datasets.make_regression(1000, 288, n_targets=1, noise=0.02)
    y = y.reshape(-1, 1)

    lreg = linear_model.LinearRegression(n_jobs=-1)
    lreg.fit(X, y)
    print (lreg.score(X, y))
    print('predict=', lreg.predict(X[829: 833]), y[829: 833] )

    model = Model()
    epochs = 100
    critreion = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=5e-5)

    dataset = Dataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch, data in enumerate(data_loader):
            x = data['data']
            y = data['target']

            yhat = model(x)
            loss = critreion(y, yhat)
            
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f'epoch[{epoch}], loss={total_loss/len(data_loader):0.4f}')
    torch.save(model, 'model.bin')


    model = torch.load('model.bin')
    print(model.parameters)

    with torch.no_grad():
        xt = (dataset[30:33]['data'])
        yt = dataset[30:33]['target']
        pred = model(xt)
        print('original', yt, 'pred', pred, 'loss', critreion(yt, pred).item())