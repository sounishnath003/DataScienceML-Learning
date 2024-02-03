from kfp.dsl import component, Input, Metrics, Model, Dataset, Output
from .constants import KubeflowConfig


@component(
    base_image=KubeflowConfig.base_image,
    packages_to_install=KubeflowConfig.packages_to_install,
    output_component_file=KubeflowConfig.save_to("evaluate-model.yaml"),
)
def evaluate_model(
    val_dfx_in: Input[Dataset],
    trained_model_in: Input[Model],
    val_metrics_out: Output[Metrics],
):
    import os
    import logging
    import pandas as pd
    import numpy as np
    import joblib

    from datetime import datetime

    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    logging.getLogger().setLevel(logging.INFO)

    val_dfx = pd.read_csv(val_dfx_in.path)
    val_x = val_dfx.drop(["quality"], axis=1)
    val_y = val_dfx[["quality"]]

    model = joblib.load(f"{trained_model_in.path}.joblib")

    logging.info("starting the model evaluation metrics")
    preds = model.predict(val_x)

    def eval_metrics(preds, actual):
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mae = mean_absolute_error(actual, preds)
        r2 = r2_score(actual, preds)

        return (rmse, mae, r2)

    (rmse, mae, r2) = eval_metrics(preds, val_y)

    val_metrics_out.log_metric("val.rmse", rmse)
    val_metrics_out.log_metric("val.mae", mae)
    val_metrics_out.log_metric("val.r2", r2)
    val_metrics_out.log_metric("trainedOn", datetime.now().strftime("%D-%M-%Y %H:%M"))

    logging.info("model evaluation has been done")
