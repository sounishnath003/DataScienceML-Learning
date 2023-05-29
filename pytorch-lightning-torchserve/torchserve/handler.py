"""
# _* coding: utf8 *_

filename: handler.py

@author: sounishnath
createdAt: 2023-05-28 12:54:03
"""

import os

import pytorch_lightning as pl
from ts.torch_handler.base_handler import BaseHandler

import torch
from torchh.dataset import ImdbDataset
from torchh.model import ImdbNeuralNet, LitImdbNeuralNet


class ImdbModelHandler(BaseHandler):
    def __init__(self):
        super(ImdbModelHandler, self).__init__()
        self._context = None
        self.initialized = False
        self._model = None
        self.device = None

    def initialize(self, context):
        self.manifest = context.manifest

        properties = context.system_properties
        self.device = "cuda" if torch.has_cuda() else "cpu"

        print(
            {
                "properties": properties,
                "device": self.device,
            }
        )

        self._model = LitImdbNeuralNet(foundation_model=ImdbNeuralNet(), n_classes=1)
        self._model.load_from_checkpoint(os.path.join(os.getcwd(), "models", "best_model.ckpt"))
        self._model.eval()
        self.initialized = True

    def preprocess(self, data):
        input_data = data[0].get("data", None)
        if not input_data:
            raise Exception(f"input_data is NONE....")

        _batch = ImdbDataset(review=[input_data.strip()], sentiment=[0])[-1]
        _batch = {
            **_batch,
            "input_ids": _batch.get("input_ids").unsqueeze(0),
            "token_type_ids": _batch.get("token_type_ids").unsqueeze(0),
            "attention_mask": _batch.get("attention_mask").unsqueeze(0),
            "sentiment": _batch["sentiment"],
        }
        return _batch

    def inference(self, data, *args, **kwargs):
        self._model.eval()
        return self._model.forward(data).detach().cpu().numpy()

    def postprocess(self, data):
        return data

    def handle(self, data, context):
        data = self.preprocess(data)
        logits = self.inference(data)
        return self.postprocess(logits)
