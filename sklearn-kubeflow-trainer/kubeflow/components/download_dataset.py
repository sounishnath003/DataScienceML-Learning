import os
from .constants import KubeflowConfig
from kfp.dsl import component, Input, Output, Dataset


@component(
    base_image=KubeflowConfig.base_image,
    packages_to_install=KubeflowConfig.packages_to_install,
    output_component_file=KubeflowConfig.save_to("download-dataset.yaml"),
)
def download_dataset(
    csv_uri: str,
    train_dfx_out: Output[Dataset],
    val_dfx_out: Output[Dataset],
):
    import logging
    import numpy as np
    import pandas as pd

    from sklearn.model_selection import train_test_split

    logging.getLogger().setLevel(logging.INFO)

    CSV_URL = csv_uri
    logging.info("received the CSV_URL = {}".format(csv_uri))

    try:
        dfx = pd.read_csv(CSV_URL, sep=";")
    except Exception as e:
        logging.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s",
            e,
        )

    train_dfx, val_dfx = train_test_split(dfx, test_size=0.30, random_state=31)

    logging.info(
        {
            "train_dfx.shape": train_dfx.shape,
            "val_dfx.shape": val_dfx.shape,
        }
    )

    train_dfx.to_csv(train_dfx_out.path, index=False)
    val_dfx.to_csv(val_dfx_out.path, index=False)
