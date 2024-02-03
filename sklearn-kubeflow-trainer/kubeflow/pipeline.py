from kfp.dsl import Input, Output, Artifact, Model, pipeline, component
from .components.download_dataset import download_dataset
from .components.train_model import train_model
from .components.evaluate_model import evaluate_model
from .components.constants import KubeflowConfig

from kfp.compiler import Compiler
from google.cloud import aiplatform
from google.cloud import aiplatform_v1

from datetime import datetime


@pipeline(
    name="sklearn-simple-kfp-trainer",
    description="a simple demonstrating on a sklearn model trainer kfp pipeline",
    pipeline_root="gs://kubeflow-out/sklearn-simple-kfp-trainer/",
)
def pipeline(
    csv_uri: str,
    alpha: float,
    l1_ratio: float,
):
    download_dataset_comp = download_dataset(csv_uri=csv_uri)
    train_model_comp = train_model(
        alpha=alpha,
        l1_ratio=l1_ratio,
        train_dfx_in=download_dataset_comp.outputs["train_dfx_out"],
    )
    eval_model_comp = evaluate_model(
        val_dfx_in=download_dataset_comp.outputs["val_dfx_out"],
        trained_model_in=train_model_comp.outputs["model_out"],
    )


if __name__ == "__main__":
    TIMESTAMP = datetime.now().strftime("%y%m%d%H%M%S")
    print("timestamp current = {}".format(TIMESTAMP))

    Compiler().compile(
        pipeline_func=pipeline,
        package_path=KubeflowConfig.save_to("sklearn-simple-kfp-trainer.yaml"),
        pipeline_parameters={
            "csv_uri": "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "alpha": 0.5,
            "l1_ratio": 0.5,
        },
    )

    with open(KubeflowConfig.save_to("sklearn-simple-kfp-trainer.yaml"), "r+") as fp:
        print(fp.read())
        fp.close()

    custom_pipeline = aiplatform.PipelineJob(
        display_name="sklearn-simple-kfp-trainer-{}".format(TIMESTAMP),
        template_path=KubeflowConfig.save_to("sklearn-simple-kfp-trainer.yaml"),
        parameter_values={
            "csv_uri": "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "alpha": 0.5,
            "l1_ratio": 0.5,
        },
        enable_caching=True,
        job_id="sklearn-simpler-kfp-trainer-{}".format(TIMESTAMP),
        location="asia-south1",
        project="sounish-cloud-workstation",
    )
    custom_pipeline.submit()
