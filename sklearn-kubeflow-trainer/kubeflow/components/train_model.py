from .constants import KubeflowConfig
from kfp.dsl import Input, Output, Dataset, component, Model, Metrics


@component(
    base_image=KubeflowConfig.base_image,
    packages_to_install=KubeflowConfig.packages_to_install,
    output_component_file=KubeflowConfig.save_to("split-dataset.yaml"),
)
def train_model(
    alpha: float,
    l1_ratio: float,
    train_dfx_in: Input[Dataset],
    model_out: Output[Model],
    train_metrics_out: Output[Metrics],
):
    import os
    import logging
    import pandas as pd

    import numpy as np
    from joblib import dump
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    logging.getLogger().setLevel(logging.INFO)

    train_dfx = pd.read_csv(train_dfx_in.path)

    train_x = train_dfx.drop(["quality"], axis=1)
    train_y = train_dfx[["quality"]]

    alpha = 0.5 if float(alpha) is None else float(alpha)
    l1_ratio = 0.5 if float(alpha) is None else float(alpha)

    logging.info("starting the model training job")
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=31)
    model.fit(train_x, train_y)
    logging.info("the model training job has been finished")

    preds = model.predict(train_x)

    def eval_metrics(preds, actual):
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mae = mean_absolute_error(actual, preds)
        r2 = r2_score(actual, preds)

        return (rmse, mae, r2)

    (rmse, mae, r2) = eval_metrics(preds, train_y)

    model_out.metadata["train_rmse"] = rmse
    model_out.metadata["train_mae"] = mae
    model_out.metadata["train_r2"] = r2

    train_metrics_out.log_metric("train.rmse", rmse)
    train_metrics_out.log_metric("train.mae", mae)
    train_metrics_out.log_metric("train.r2", r2)

    dump(model, filename=f"{model_out.path}.joblib")
    logging.info("training metrics has been written...")
