import os
from kfp.dsl import (
    Input,
    Output,
    component,
    Artifact,
    Model,
    Metrics,
)

from .constants import KubeflowConfiguration


@component(
    base_image="gcr.io/deeplearning-platform-release/pytorch-gpu.2-0.py310",
    output_component_file=os.path.join(
        KubeflowConfiguration.ARRTIFACT_DIRECTORY,
        "components",
        "train-model.component.yaml",
    ),
    packages_to_install=[
        "transformers",
        "lightning",
        "torchmetrics",
        "scikit-learn",
        "black",
        "datasets",
        "torchserve",
        "torch-model-archiver",
    ],
)
def train_model(
    dataset: Input[Artifact],
    lit_logger_out: Output[Artifact],
    model_checkpoint_out: Output[Artifact],
    lit_model_out: Output[Model],
    metrics_out: Output[Metrics],
):
    from datasets import load_dataset
    import os
    import sys
    import tarfile
    import time

    import numpy as np
    import pandas as pd
    from packaging import version
    from torch.utils.data import Dataset
    from tqdm import tqdm
    import urllib

    from transformers import AutoTokenizer

    def reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = progress_size / (1024.0**2 * duration)
        percent = count * block_size * 100.0 / total_size

        sys.stdout.write(
            f"\r{int(percent)}% | {progress_size / (1024.**2):.2f} MB "
            f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
        )
        sys.stdout.flush()

    def download_dataset():
        source = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        target = "aclImdb_v1.tar.gz"

        if os.path.exists(target):
            os.remove(target)

        if not os.path.isdir("aclImdb") and not os.path.isfile("aclImdb_v1.tar.gz"):
            urllib.request.urlretrieve(source, target, reporthook)

        if not os.path.isdir("aclImdb"):
            with tarfile.open(target, "r:gz") as tar:
                tar.extractall()

    def load_dataset_into_to_dataframe():
        basepath = "aclImdb"

        labels = {"pos": 1, "neg": 0}

        df = pd.DataFrame()

        with tqdm(total=50000) as pbar:
            for s in ("test", "train"):
                for l in ("pos", "neg"):
                    path = os.path.join(basepath, s, l)
                    for file in sorted(os.listdir(path)):
                        with open(
                            os.path.join(path, file), "r", encoding="utf-8"
                        ) as infile:
                            txt = infile.read()

                        if version.parse(pd.__version__) >= version.parse("1.3.2"):
                            x = pd.DataFrame(
                                [[txt, labels[l]]], columns=["review", "sentiment"]
                            )
                            df = pd.concat([df, x], ignore_index=False)

                        else:
                            df = df.append([[txt, labels[l]]], ignore_index=True)
                        pbar.update()
        df.columns = ["text", "label"]

        np.random.seed(0)
        df = df.reindex(np.random.permutation(df.index))

        print("Class distribution:")
        np.bincount(df["label"].values)

        return df

    def partition_dataset(df):
        df_shuffled = df.sample(frac=1, random_state=1).reset_index()

        df_train = df_shuffled.iloc[:35_000]
        df_val = df_shuffled.iloc[35_000:40_000]
        df_test = df_shuffled.iloc[40_000:]

        df_train.to_csv("train.csv", index=False, encoding="utf-8")
        df_val.to_csv("val.csv", index=False, encoding="utf-8")
        df_test.to_csv("test.csv", index=False, encoding="utf-8")

    def tokenize_text(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    download_dataset()
    dfx = load_dataset_into_to_dataframe()

    if not (
        os.path.exists("train.csv")
        and os.path.exists("val.csv")
        and os.path.exists("test.csv")
    ):
        partition_dataset(dfx)

    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": "train.csv",
            "validation": "val.csv",
            "test": "test.csv",
        },
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("Tokenizer input max length:", tokenizer.model_max_length, flush=True)
    print("Tokenizer vocabulary size:", tokenizer.vocab_size, flush=True)

    print("Tokenizing ...", flush=True)
    imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
    del imdb_dataset
    imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # imdb_tokenized.save_to_disk(dataset_out.path)
    # print("dataset has been tokenized and saved inside {}".format(dataset_out.path))

    from datasets import load_dataset
    from torch.utils.data import Dataset, DataLoader

    import os
    import torch
    import torch.nn as nn
    import torchmetrics
    from lightning import pytorch as pl
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        AutoModel,
    )

    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    class LightningModel(pl.LightningModule):
        def __init__(self, model, learning_rate=5e-5):
            super().__init__()

            self.learning_rate = learning_rate
            self.pretrained_model = model
            self.classifier = nn.Sequential(
                nn.Linear(768, 768), nn.ReLU(), nn.Dropout(0.20), nn.Linear(768, 2)
            )

            self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

        def forward(self, input_ids, attention_mask, labels):
            hidden_logits = self.pretrained_model.forward(
                input_ids, attention_mask=attention_mask
            )[0]
            pooled_output = hidden_logits[:, 0]
            logits = self.classifier.forward(pooled_output)
            loss = None

            if not labels is None:
                loss = nn.CrossEntropyLoss()(
                    logits.to(self.device), labels.view(-1).to(self.device)
                )

            return {"logits": logits, "loss": loss}

        def training_step(self, batch, batch_idx):
            outputs = self(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )
            self.log("train_loss", outputs["loss"])
            with torch.no_grad():
                logits = outputs["logits"]
                predicted_labels = torch.argmax(logits, 1)
                acc = self.train_acc(predicted_labels, batch["label"])
                self.log("train_acc", acc, on_epoch=True, on_step=False)
                self.log_dict(
                    {"loss": outputs["loss"], "train_acc": acc},
                    on_epoch=True,
                    on_step=False,
                )
            return outputs["loss"]  # this is passed to the optimizer for training

        def validation_step(self, batch, batch_idx):
            outputs = self(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )
            self.log("val_loss", outputs["loss"], prog_bar=True)

            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
            acc = self.val_acc(predicted_labels, batch["label"])
            self.log_dict(
                {"loss": outputs["loss"], "train_acc": acc},
                on_epoch=True,
                on_step=False,
            )
            self.log("val_acc", acc, prog_bar=True)

        def test_step(self, batch, batch_idx):
            outputs = self(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )

            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
            acc = self.test_acc(predicted_labels, batch["label"])
            self.log("accuracy", acc, prog_bar=True)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                self.trainer.model.parameters(), lr=self.learning_rate
            )
            return optimizer

    class IMDBDataset(Dataset):
        def __init__(self, dataset_dict, partition_key="train"):
            self.partition = dataset_dict[partition_key]

        def __getitem__(self, index):
            return self.partition[index]

        def __len__(self):
            return self.partition.num_rows

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    print(
        {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
        }
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=3,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=12,
        num_workers=3,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers=3,
        drop_last=True,
    )

    model = AutoModel.from_pretrained("distilbert-base-uncased")

    lit_model = LightningModel(model)
    print(lit_model)

    callbacks = [
        ModelCheckpoint(
            dirpath=model_checkpoint_out.path,
            save_top_k=1,
            mode="max",
            monitor="val_acc",
            verbose=True,
            save_on_train_epoch_end=True,
        )  # save top 1 model
    ]
    logger = CSVLogger(save_dir=lit_logger_out.path, name="my-model")

    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        deterministic=True,
        enable_progress_bar=True
    )

    trainer.fit(
        lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    test_acc = trainer.test(lit_model, dataloaders=test_dataloader, verbose=True)[0]["accuracy"]
    val_acc = trainer.test(lit_model, dataloaders=val_dataloader, verbose=True)[0]["accuracy"]
    print(test_acc)

    metrics_out.metadata["val_acc"] = val_acc
    metrics_out.metadata["test_acc"] = test_acc
    metrics_out.metadata["best_ckpt"] = trainer.ckpt_path

    metrics_out.log_metric("val_acc", val_acc)
    metrics_out.log_metric("test_acc", test_acc)

    trainer.save_checkpoint(lit_model_out.path, weights_only=True)