"""
# _* coding: utf8 *_

filename: dataset.py

@author: sounishnath
createdAt: 2022-07-21 20:32:29
"""

import torch


class Dataset:
    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return {
            "data": torch.tensor(self.data[item], dtype=torch.float),
            "target": torch.tensor(self.targets[item], dtype=torch.long),
        }
