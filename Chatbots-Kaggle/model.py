"""
# _* coding: utf8 *_

filename: model.py

@author: sounishnath
createdAt: 2022-12-13 21:52:25
"""

import torch.nn as nn

import torch


class IntentDLModel(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        n_embedding_dim: int,
        padding_idx: int,
        n_hidden_layer: int,
        n_hidden_layer_neurons: int,
        n_classes: int,
    ) -> None:
        super(IntentDLModel, self).__init__()
        self.embedding_layer = nn.Sequential(
            nn.Embedding(
                num_embeddings=n_embeddings,
                embedding_dim=n_embedding_dim,
                padding_idx=padding_idx,
            ),
            nn.Dropout(0.10, inplace=True),
            nn.Linear(in_features=n_embedding_dim, out_features=n_hidden_layer_neurons),
            nn.ReLU(inplace=True),
        )
        self.linear_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 9216),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.05),
                )
                for _ in range(n_hidden_layer)
            ]
        )
        self.classifier = nn.Linear(9216, n_classes)

    def forward(self, token_type_ids, tag=None):
        out = self.embedding_layer(token_type_ids)
        out = out.view(-1, out.size(1) * out.size(2))
        out = self.classifier(out)
        return out


class IntentRnnModel(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        n_embedding_dim: int,
        padding_idx: int,
        n_hidden_layer: int,
        n_hidden_layer_neurons: int,
        n_classes: int,
    ) -> None:
        super(IntentRnnModel, self).__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=n_embeddings,
            embedding_dim=n_embedding_dim,
            padding_idx=padding_idx,
        )
        self.lstm_layer = nn.LSTM(
            input_size=n_embedding_dim,
            hidden_size=n_hidden_layer_neurons,
            batch_first=True,
            dropout=0.10,
        )
        self.dense_layer = nn.Sequential(
            nn.Linear(in_features=9216, out_features=1024),
            nn.Dropout(0.5, inplace=True),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, token_type_ids, tag=None):
        embeds = self.embedding_layer(token_type_ids)
        lstm_out, hidden = self.lstm_layer(embeds)
        out = self.dense_layer(
            torch.clone(lstm_out.reshape(-1, lstm_out.size(1) * lstm_out.size(2)))
        )
        out = self.classifier(torch.clone(out))
        return out
