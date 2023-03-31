"""
# _* coding: utf8 *_

filename: config.py

@author: sounishnath
createdAt: 2023-03-29 21:37:01
"""

import os
from dataclasses import dataclass


@dataclass
class Configuration:
    dataset_path = os.path.join(os.getcwd(), "data", "data.txt")
    batch_size = 64
    block_size = 256  # maximum length of generating tokens
    epochs = 999
    eval_intervals = 5
    eval_iters = 200
    eval_iteration = 50
    learning_rate = 3e-4
    device = "cpu"
    embedding_size = 384
    n_attention_heads = 6
    n_layers = 6
    dropout = 0.20
