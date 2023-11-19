from almostanytextclassifier.utils import Logger, IMDBDataset, ImdbDatasetUtility
from lightning import pytorch as pl
import os
from datasets import load_dataset

from almostanytextclassifier.model import AlmostAnyTextClassifierLitModel

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch.utils.data import DataLoader
from torch import nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


if __name__ == "__main__":
    pl.seed_everything(101)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    __MODEL_NAME__ = "distilbert-base-uncased"

    imdb_dataset_utility = ImdbDatasetUtility()
    # imdb_dataset_utility.download_dataset()
    dfx = imdb_dataset_utility.load_dataset_into_to_dataframe()
    if not (
        os.path.exists("train.csv")
        and os.path.exists("val.csv")
        and os.path.exists("test.csv")
    ):
        imdb_dataset_utility.partition_dataset(dfx)

    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv",
        },
    )
    Logger.get_logger().info(imdb_dataset)

    tokenizer = AutoTokenizer.from_pretrained(__MODEL_NAME__)
    imdb_tokenized = imdb_dataset.map(
        lambda data: tokenizer(data["text"], padding=True, truncation=True),
        batched=True,
        batch_size=None,
    )
    del imdb_dataset
    imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    val_dataset = IMDBDataset(imdb_tokenized, partition_key="val")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")
    Logger.get_logger().debug(test_dataset[0])

    train_dataloader = DataLoader(
        train_dataset, batch_size=12, shuffle=True, num_workers=1, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=12, shuffle=False, num_workers=1, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=12, shuffle=False, num_workers=1, drop_last=True
    )

    pretrained_hf_model = AutoModelForSequenceClassification.from_pretrained(
        __MODEL_NAME__, num_labels=2
    )
    # not supported for mac m1
    # pretrained_hf_model = torch.compile(pretrained_hf_model)

    lit_model = AlmostAnyTextClassifierLitModel(
        model=pretrained_hf_model,
        num_classes=2,
    )
    Logger.get_logger().info(lit_model)
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc"),
        EarlyStopping(monitor="val_loss", verbose=True),
    ]
    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=3,
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
    trainer.fit(lit_model, train_dataloader, val_dataloader)

    trainer.test(lit_model, test_dataloader, ckpt_path="best")
