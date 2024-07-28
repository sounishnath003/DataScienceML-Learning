# * coding utf-8 *
# @author: @github/sounishnath003
# createdAt: 25-07-2024

import argparse
import json

from loguru import logger
from sklearn import model_selection
from lightning import pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.calibration import LabelEncoder

from torch.utils.data import DataLoader
from transformers import DefaultDataCollator

from flywheel.classifier import TextClassifierLitModel
from flywheel.dataset import TextDataset, Tokenizer, read_dataset

if __name__ == "__main__":
    pl.seed_everything(42)

    parser = argparse.ArgumentParser(add_help=True)
    # download following dataset from kaggle OSS
    # https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset/
    # store it in `data` directory
    parser.add_argument("--dataset", required=True, type=str) # https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset/
    parser.add_argument("--batch_size", required=False, type=int, default=72)
    parser.add_argument("--epochs", required=False, type=int, default=10)
    parser.add_argument("--lr", required=False, type=float, default=3e-4)

    opts, unknown_opts = parser.parse_known_args()
    logger.info("args: {}, unknown_args: {}", opts, unknown_opts)

    # read the dataset from the dataset path from runtime args
    dfx = read_dataset(opts.dataset, text_column="Document", label_column="Topic_group")
    logger.info("sample records: {}", dfx.sample(3))

    # do label encoding of the label column
    label_encoder = LabelEncoder()
    dfx.loc[:, "label_encoded"] = label_encoder.fit_transform(
        dfx.loc[:, "label"].values
    )
    label_2_ids = {index: cls for index, cls in enumerate(label_encoder.classes_)}
    open("labels.json", "w+").write(json.dumps(label_2_ids))
    logger.info("labels.json has been written and saved to disk")

    # adding the stratified kfold
    dfx.loc[:, "fold"] = -1
    logger.info("performing the stratified kfold splits into dfx dataset")
    skf = model_selection.StratifiedKFold(shuffle=True)
    for fold, (trn_, val_) in enumerate(
        skf.split(X=dfx.loc[:, "text"].values, y=dfx.loc[:, "label_encoded"].values)
    ):
        dfx.loc[val_, "fold"] = fold
    logger.info("sample records: {}", dfx.sample(3))

    train_dfx, valid_dfx = dfx[dfx["fold"] != 0].copy(), dfx[dfx["fold"] == 0].copy()
    logger.info("train.dfx.shape: {}", train_dfx.shape)
    logger.info("valid.dfx.shape: {}", valid_dfx.shape)

    train_dataset = TextDataset(
        texts=train_dfx.text.values, labels=train_dfx.label_encoded.values
    )
    valid_dataset = TextDataset(
        texts=valid_dfx.text.values, labels=valid_dfx.label_encoded.values
    )

    logger.debug(train_dataset[3])

    # convert the datasets into torch data loaders
    train_dataloader = DataLoader(
        num_workers=2,
        dataset=train_dataset,
        batch_size=opts.batch_size,
        collate_fn=DefaultDataCollator(),
        shuffle=False,
        drop_last=True,
        persistent_workers=True,
    )
    valid_dataloader = DataLoader(
        num_workers=1,
        dataset=valid_dataset,
        batch_size=opts.batch_size,
        collate_fn=DefaultDataCollator(),
        shuffle=False,
        drop_last=True,
        persistent_workers=True,
    )

    trainer = pl.Trainer(
        max_epochs=opts.epochs,
        fast_dev_run=False,
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        logger=MLFlowLogger(
            experiment_name="lightning_logs", tracking_uri="file:ml-runs"
        ),
    )

    lit_model = TextClassifierLitModel(
        num_classes=label_encoder.classes_.size, lr=opts.lr
    )
    logger.info("model.information: {}", lit_model)

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
