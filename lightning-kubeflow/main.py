"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-08-20 12:13:42

It's a sample Imdb dataset movie review classifier built using lightning pytorch and kubeflow to MlOps solutions
which is ported to run locally on any computer.

"""

from dataclasses import dataclass


@dataclass
class Output:
    path: str = ""


def getLogger(logger_name: str = "LightningKubeFlowLogger"):
    import logging

    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(logger_name)


def download_dataset(gcs_file_path: str, output: Output):
    import pandas as pd
    from sklearn import model_selection

    dfx = pd.read_csv(gcs_file_path)
    dfx["kfold"] = -1
    skf = model_selection.StratifiedKFold(shuffle=True, random_state=111)
    for fold, (_, valid_) in enumerate(
        skf.split(X=dfx["review"].values, y=dfx["sentiment"].values)
    ):
        dfx.loc[valid_, "kfold"] = fold
    print(dfx.sample(5))

    with open(output.path, "w") as f:
        dfx.to_csv(f, index=False, header=True)
        f.close()


def split_dataset_into_train_valid(
    dataset_path: Output,
    train_dfx_output: Output,
    valid_dfx_output: Output,
):
    import pandas as pd

    dfx = pd.read_csv(dataset_path.path)
    K_FOLD: int = 0

    train_dfx = dfx[dfx.loc[:, "kfold"] != K_FOLD]
    val_dfx = dfx[dfx.loc[:, "kfold"] == K_FOLD]

    with open(train_dfx_output.path, "w") as f:
        train_dfx.to_csv(f, index=False, header=True)
        f.close()

    with open(valid_dfx_output.path, "w") as f:
        val_dfx.to_csv(f, index=False, header=True)
        f.close()


def download_pretrained_toknizer(model_name: str, tokenizer_output: Output):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tokenizer_output.path)


def download_pretrained_hf_model(model_name: str, pretrained_model_output: Output):
    from transformers import AutoModel

    hf_model = AutoModel.from_pretrained("bert-base-uncased")
    hf_model.save_pretrained(pretrained_model_output.path)


def finetune_deep_neural_network(
    tokenizer: Output,
    train_dfx_path: Output,
    valid_dfx_path: Output,
    hf_model_in: Output,
    lightning_finetune_model_out: Output,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from lightning import pytorch as pl
    import torchmetrics as tmetrics

    from typing import Any, Optional
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from transformers import BertConfig, AutoModel

    import os
    import pandas as pd
    from transformers import AutoTokenizer

    from torch.utils.data import DataLoader
    from lightning.pytorch.callbacks import ModelCheckpoint

    class ImdbDeepNeuralNetworkLitModel(pl.LightningModule):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super(ImdbDeepNeuralNetworkLitModel, self).__init__()

            self.num_labels: int = 2

            self.bert_config = BertConfig(
                hidden_dropout_prob=0.20,
                layer_norm_eps=1e-7,
                dropout=0.20,
                max_position_embeddings=1536,
                ignore_mismatched_sizes=True,
            )
            self.bert_model = AutoModel.from_pretrained(
                "bert-base-uncased",
                config=self.bert_config,
                ignore_mismatched_sizes=True,
            )
            self.pre_classifier = nn.Linear(
                self.bert_config.hidden_size, self.bert_config.hidden_size
            )
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.20)
            self.classifier = nn.Linear(self.bert_config.hidden_size, self.num_labels)

            self.valid_metrics = []

        def common_step(self, input_ids, attention_mask, token_type_ids):
            pooled_output = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).pooler_output
            logits = self.pre_classifier.forward(pooled_output)
            logits = self.relu.forward(logits)
            logits = self.dropout.forward(logits)
            logits = self.classifier.forward(logits)

            return logits

        def compute_loss(self, preds, targets):
            loss = nn.CrossEntropyLoss()(
                preds.view(-1, self.num_labels), targets.view(-1)
            )
            return loss

        def forward(self, input_ids, attention_mask, token_type_ids):
            logits = self.common_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return logits

        def training_step(
            self, batch, batch_idx, *args: Any, **kwargs: Any
        ) -> STEP_OUTPUT:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            targets = batch["target"]

            logits = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = self.compute_loss(logits, targets)
            self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)

            return loss

        def validation_step(
            self, batch, batch_idx, *args: Any, **kwargs: Any
        ) -> STEP_OUTPUT | None:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            targets = batch["target"]

            logits = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = self.compute_loss(logits, targets)
            self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

            preds = torch.argmax(logits, dim=1)
            val_acc = tmetrics.Accuracy(task="multiclass", num_classes=2)(
                preds.detach().cpu(), targets.detach().cpu()
            )
            self.log("val_acc", val_acc, prog_bar=True, on_step=True, on_epoch=True)
            self.valid_metrics.append({"val_loss": loss, "val_acc": val_acc})

            return {"val_loss": loss, "val_acc": val_acc}

        def on_validation_end(self) -> None:
            avg_loss = torch.stack([x["val_loss"] for x in self.valid_metrics]).mean()
            avg_val_acc = torch.stack([x["val_acc"] for x in self.valid_metrics]).mean()

            getLogger(ImdbDeepNeuralNetworkLitModel.__name__).info(
                f"val_loss: {avg_loss}"
            )
            getLogger(ImdbDeepNeuralNetworkLitModel.__name__).info(
                f"val_acc: {avg_val_acc}"
            )
            self.valid_metrics.clear()
            self.valid_metrics = []

            return {"val_loss": avg_loss, "val_acc": avg_val_acc}

        def predict_step(
            self, batch: Any, batch_idx: int, dataloader_idx: int = 0
        ) -> Any:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]

            logits = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            return logits

        def configure_optimizers(self) -> Any:
            opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
            sch1 = torch.optim.lr_scheduler.StepLR(opt, step_size=15, verbose=True)
            sch2 = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=1e-3, total_steps=2000
            )

            return [opt], [sch1, sch2]

    class ImdbDataset:
        def __init__(self, reviews, sentiments) -> None:
            self.reviews = reviews
            self.sentiments = sentiments

            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        def __len__(self):
            return len(self.reviews)

        def __getitem__(self, item):
            review = self.reviews[item]
            tokenized_review = self.tokenizer.encode_plus(
                review,
                None,
                add_special_tokens=True,
                padding="max_length",
                max_length=1301,
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
            )
            sentiment = 1 if self.sentiments[item] == "positive" else 0
            return {
                "input_ids": self.to_tensor(
                    tokenized_review.get("input_ids"), torch.long
                ),
                "token_type_ids": self.to_tensor(
                    tokenized_review.get("token_type_ids"), torch.long
                ),
                "attention_mask": self.to_tensor(
                    tokenized_review.get("attention_mask"), torch.long
                ),
                "target": self.to_tensor(sentiment, torch.long),
            }

        def to_tensor(self, data, _dtype):
            return torch.tensor(data, dtype=_dtype)

    train_dfx = pd.read_csv(train_dfx_path.path)
    valid_dfx = pd.read_csv(valid_dfx_path.path)

    train_dataset = ImdbDataset(
        train_dfx.loc[:, "review"].values, train_dfx.loc[:, "sentiment"].values
    )
    valid_dataset = ImdbDataset(
        valid_dfx.loc[:, "review"].values, valid_dfx.loc[:, "sentiment"].values
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=os.cpu_count() or 2,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=os.cpu_count() or 2,
        drop_last=True,
    )

    lit_model = ImdbDeepNeuralNetworkLitModel()
    getLogger().info(lit_model)
    lit_trainer = pl.Trainer(
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        max_epochs=10,
        default_root_dir=lightning_finetune_model_out.path,
        callbacks=[
            ModelCheckpoint(
                lightning_finetune_model_out.path,
                verbose=True,
                save_top_k=3,
                monitor="val_loss",
            )
        ],
        fast_dev_run=False,
    )
    lit_trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    


if __name__ == "__main__":
    import os

    logger = getLogger()

    download_dataset(
        gcs_file_path="gs://workspace-bucket-01/IMDB Dataset.csv",
        output=Output(path=os.path.join(os.getcwd(), "jobs", "dataset")),
    )
    split_dataset_into_train_valid(
        dataset_path=Output(path=os.path.join(os.getcwd(), "jobs", "dataset")),
        train_dfx_output=Output(path=os.path.join(os.getcwd(), "jobs", "train_dfx")),
        valid_dfx_output=Output(path=os.path.join(os.getcwd(), "jobs", "valid_dfx")),
    )
    download_pretrained_toknizer(
        model_name="bert-base-uncased",
        tokenizer_output=Output(path=os.path.join(os.getcwd(), "jobs", "tokenizer")),
    )
    download_pretrained_hf_model(
        model_name="bert-base-uncased",
        pretrained_model_output=Output(
            path=os.path.join(os.getcwd(), "jobs", "pretrained_hf_model")
        ),
    )
    finetune_deep_neural_network(
        tokenizer=Output(path=os.path.join(os.getcwd(), "jobs", "tokenizer")),
        train_dfx_path=Output(path=os.path.join(os.getcwd(), "jobs", "train_dfx")),
        valid_dfx_path=Output(path=os.path.join(os.getcwd(), "jobs", "valid_dfx")),
        hf_model_in=Output(
            path=os.path.join(os.getcwd(), "jobs", "pretrained_hf_model")
        ),
        lightning_finetune_model_out=Output(
            path=os.path.join(os.getcwd(), "jobs", "finetuned_clf_model")
        ),
    )
