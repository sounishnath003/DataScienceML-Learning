"""
# _* coding: utf8 *_

filename: preranking_model.py

@author: sounishnath
createdAt: 2023-10-28 20:02:18
"""


from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import pytorch as pl


class LightSqueezeExicitationModel(nn.Module):
    def __init__(
        self, field_size: int, embedding_dim: int = 768, *args, **kwargs
    ) -> None:
        # part of IntTower Pre-ranking Model Research paper. ...
        super(LightSqueezeExicitationModel, self).__init__(*args, **kwargs)
        self.softmax = nn.Softmax(dim=1)
        self.field_size = field_size
        self.embedding_dim = embedding_dim
        self.excitation_layer = nn.Linear(self.field_size, self.field_size, bias=False)

    def forward(self, X):
        A = self.excitation_layer.forward(X)
        # A = self.softmax.forward(A)
        # print(torch.mean(A, dim=1).unsqueeze(dim=0).T.size())
        return X * torch.mean(A, dim=1).unsqueeze(dim=0).T


class SqueezeExcitationModel(nn.Module):
    def __init__(self, filed_size, reduction_ratio=3, *args, **kwargs) -> None:
        # part of IntTower Pre-ranking Model Research paper. ...
        super(SqueezeExcitationModel, self).__init__(*args, **kwargs)
        self.filed_size = filed_size
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation_layer = nn.Sequential(
            nn.Linear(filed_size, reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(reduction_ratio, filed_size, bias=False),
            nn.ReLU(),
        )

    def forward(self, X):
        Z = torch.mean(X, dim=-1, out=None)
        A = self.excitation(Z)
        V = torch.mul(X, torch.unsqueeze(A, dim=2))
        return X + V


##########################################################
#### https://github.com/coaxsoft/pytorch_bert/ -
# easy implementation by using pytorch API implementation
# good to use the implement ... if you want to train a transformer BERT
# from scratch ........... Kudos COAX
##########################################################


class JointEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, size):
        super(JointEmbeddingLayer, self).__init__()

        self.size = size

        self.token_emb = nn.Embedding(vocab_size, size)
        self.segment_emb = nn.Embedding(vocab_size, size)

        self.norm = nn.LayerNorm(size)

    def forward(self, input_tensor):
        sentence_size = input_tensor.size(-1)
        pos_tensor = self.attention_position(self.size, input_tensor)

        segment_tensor = torch.zeros_like(input_tensor).to(device)
        segment_tensor[:, sentence_size // 2 + 1 :] = 1

        output = (
            self.token_emb(input_tensor) + self.segment_emb(segment_tensor) + pos_tensor
        )
        return self.norm(output)

    def attention_position(self, dim, input_tensor):
        batch_size = input_tensor.size(0)
        sentence_size = input_tensor.size(-1)

        pos = torch.arange(sentence_size, dtype=torch.long).to(device)
        d = torch.arange(dim, dtype=torch.long).to(device)
        d = 2 * d / dim

        pos = pos.unsqueeze(1)
        pos = pos / (1e4**d)

        pos[:, ::2] = torch.sin(pos[:, ::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        return pos.expand(batch_size, *pos.size())

    def numeric_position(self, dim, input_tensor):
        pos_tensor = torch.arange(dim, dtype=torch.long).to(device)
        return pos_tensor.expand_as(input_tensor)


class AttentionHead(nn.Module):
    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()

        self.dim_inp = dim_inp

        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        query, key, value = (
            self.q(input_tensor),
            self.k(input_tensor),
            self.v(input_tensor),
        )

        scale = query.size(1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale

        scores = scores.masked_fill_(attention_mask, -1e9)
        attn = f.softmax(scores, dim=-1)
        context = torch.bmm(attn, value)

        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_inp, dim_out):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList(
            [AttentionHead(dim_inp, dim_out) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        s = [head(input_tensor, attention_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)


class Encoder(nn.Module):
    def __init__(self, dim_inp, dim_out, attention_heads=4, dropout=0.1):
        super(Encoder, self).__init__()

        self.attention = MultiHeadAttention(
            attention_heads, dim_inp, dim_out
        )  # batch_size x sentence size x dim_inp
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_inp, dim_out),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, dim_inp),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        context = self.attention(input_tensor, attention_mask)
        res = self.feed_forward(context)
        return self.norm(res)


class BERT(nn.Module):
    def __init__(self, vocab_size, dim_inp, dim_out, attention_heads=4):
        super(BERT, self).__init__()

        self.embedding = JointEmbeddingLayer(vocab_size, dim_inp)
        self.encoder = Encoder(dim_inp, dim_out, attention_heads)

        self.token_prediction_layer = nn.Linear(dim_inp, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.classification_layer = nn.Linear(dim_inp, 2)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        embedded = self.embedding(input_tensor)
        encoded = self.encoder(embedded, attention_mask)

        token_predictions = self.token_prediction_layer(encoded)

        first_word = encoded[:, 0, :]
        return self.softmax(token_predictions), self.classification_layer(first_word)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        n_classes: int,
        attention_heads: int = 4,
        dropout: float = 0.25,
        *args,
        **kwargs
    ) -> None:
        super(EncoderLayer, self).__init__(*args, **kwargs)
        self.attn = nn.MultiheadAttention(input_dim, 4, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(out_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        contexts_ = self.attn.forward(input_tensor, attention_mask, input_tensor)


class BidirectionEncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_dim: int,
        out_dim: int,
        n_classes: int,
        attention_heads: int = 4,
        *args,
        **kwargs
    ) -> None:
        super(BidirectionEncoderDecoderTransformer, self).__init__(*args, **kwargs)
        self.embedding = JointEmbeddingLayer(vocab_size, input_dim)
        self.encoder = EncoderLayer(input_dim, out_dim, attention_heads)
        self.token_prediction_layer = nn.Linear(input_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        embed_logits = self.embedding.forward(input_tensor)
        encoded_logits = self.encoder.forward(embed_logits, attention_mask)
        token_pred_logits = self.token_prediction_layer.forward(encoded_logits)
        first_word_ = encoded_logits[:, 0, :]
        return self.softmax.forward(token_pred_logits), self.classifier.forward(
            first_word_
        )
