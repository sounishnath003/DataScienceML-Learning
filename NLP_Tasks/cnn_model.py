

import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNet(nn.Module):
    def __init__(self, num_embedding=None, embedding_dim=None) -> None:
        super(CNNNet, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_embedding, self.embedding_dim, padding_idx=0)
        self.conv_1 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=32, kernel_size=2)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=6272, out_features=1)

    def forward(self, X):
        x = self.embedding(X)
        # print ("embedding:", x.size())
        x = self.conv_1(x)
        # print ("conv1:", x.size())
        x = self.maxpool_1(x)
        # print ("maxpool1", x.size())
        x = self.conv_2(x)
        # print ("conv2:", x.size())
        x = self.maxpool_2(x)
        # print ("maxpool2:", x.size())
        x = self.flatten(x)
        # print ("flatten:", x.size())
        logits = self.fc(x)
        # logits = logits.squeeze()
        # print ("Logits size:", logits.size())
        return torch.sigmoid(logits).squeeze()