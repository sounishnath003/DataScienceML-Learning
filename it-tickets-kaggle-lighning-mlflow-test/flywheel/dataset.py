# * coding utf-8 *
# @author: @github/sounishnath003
# createdAt: 25-07-2024


import os
import typing
from numpy import dtype
import pandas
from loguru import logger
import torch
from transformers import DistilBertTokenizer


def read_dataset(dataset_path: str, text_column: str, label_column: str):
    """reads and returns a pandas dataframe from the dataset path"""

    # determine the file extension
    file_extension = dataset_path.split(".")[-1]
    assert file_extension in [
        "csv",
        "xlsx",
    ], "filetype is not supported. only supported file types  are ['csv', 'xlsx']"
    assert True == os.path.exists(dataset_path), "file does not exists."

    # read data frame
    if file_extension == "csv":
        dfx = pandas.read_csv(dataset_path)
    else:
        dfx = pandas.read_excel(dataset_path)

    logger.info("total rows found: {}", dfx.shape[0])
    # renaming the columns
    dfx.rename({text_column: "text", label_column: "label"}, inplace=True, axis=1)

    return dfx


Tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


class TextDataset:
    def __init__(self, texts: typing.List[str], labels: typing.List[int]) -> None:
        self.texts = texts
        self.labels = labels

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        logger.info("tokenizer has been initialized.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item: int):
        """returns the index text dataset"""
        text = self.texts[item]
        label = self.labels[item]

        tokenized = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=201,
            return_attention_mask=True,
        )

        return {
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                tokenized["attention_mask"], dtype=torch.long
            ),
            "label": torch.tensor(label, dtype=torch.long),
        }
