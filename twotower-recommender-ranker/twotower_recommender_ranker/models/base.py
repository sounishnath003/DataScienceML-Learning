"""
# _* coding: utf8 *_

filename: base.py

@author: sounishnath
createdAt: 2023-10-28 19:56:57
"""

import torch
import torch.nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from abc import ABC, abstractmethod