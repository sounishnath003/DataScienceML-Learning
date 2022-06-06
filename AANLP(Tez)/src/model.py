"""
# _* coding: utf8 *_

filename: model.py

@author: sounishnath
createdAt: 2022-06-06 18:27:50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from transformers import AutoConfig, AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import config


class TextClassifierModel(nn.Module):
    def __init__(self, model_name, num_train_steps, learning_rate, num_classes) -> None:
        super(TextClassifierModel, self).__init__()
        self.num_train_steps=num_train_steps
        self.learning_rate=learning_rate
        hidden_dropout_prob=0.1
        layer_normalize_eps=1e-7
        config=AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_normalize_eps,
                "add_pooling_layer": False,
                "num_labels": 2,
            }
        )

        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output = nn.Linear(config.hidden_size, num_classes)
        self.best_accuracy=0.0

    def optimizer_scheduler(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = torch.optim.AdamW(optimizer_parameters, lr=self.learning_rate)
        sch = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps,
        )

        return opt, sch

    def loss(self, y_out, targets):
        if targets is None:
            return None
        return nn.CrossEntropyLoss()(y_out, targets)

    def monitor_metrics(self, y_outs, targets):
        if targets is None:
            return {}
        targets=targets.cpu().detach().numpy()
        y_outs=y_outs.cpu().detach().numpy().argmax(axis=1)
        accuracy=metrics.accuracy_score(y_true=targets, y_pred=y_outs)
        f1_macro=metrics.f1_score(y_true=targets, y_pred=y_outs, average='macro')
        return dict(
            accuracy=torch.tensor(accuracy, device=config.DEVICE),
            f1_macro=torch.tensor(f1_macro, device=config.DEVICE),
        )

    def forward(self, ids, attention_mask, token_type_ids, target=None):
        transformer_out = self.transformer(
            ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        out = transformer_out.pooler_output
        out = self.dropout(out)
        output = self.output(out)
        loss = self.loss(output, target)
        compute_metrics = self.monitor_metrics(output, target.view(-1,1))

        with torch.no_grad():
            cacc=compute_metrics.get('accuracy')
            if cacc > self.best_accuracy:
                self.best_accuracy=cacc
                torch.save(self.state_dict(), config.MODEL_PATH)
                print(f'new model.bin saved with accuracy: {self.best_accuracy:0.3f}')

        return output, loss, compute_metrics
        