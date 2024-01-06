import os
from kfp.dsl import pipeline
from kfp.compiler import Compiler

from .components.constants import KubeflowConfiguration
from .components.generate_dataset import generate_dataset
from .components.setup_dataloader import setup_dataloaders
from .components.train_model import train_model


@pipeline(
    name="kubefloww-trainer",
    description="a kubeflow description",
    display_name="Kubeflow Trainer",
    pipeline_root=os.path.join(
        KubeflowConfiguration.ARRTIFACT_DIRECTORY, "create-pipeline.yaml"
    ),
)
def create_pipeline():
    dataset_op = generate_dataset()
    dataloaders_op = setup_dataloaders(dataset=dataset_op.outputs["dataset_out"])
    train_model(dataset=dataset_op.outputs["dataset_out"]).set_cpu_limit(
        "16"
    ).set_memory_limit("48Gi")


if __name__ == "__main__":
    Compiler().compile(
        pipeline_func=create_pipeline,
        package_path=os.path.join(
            KubeflowConfiguration.ARRTIFACT_DIRECTORY, "create-pipeline.yaml"
        ),
    )

    with open(
        os.path.join(KubeflowConfiguration.ARRTIFACT_DIRECTORY, "create-pipeline.yaml"),
        "r+",
    ) as fp:
        print(fp.read())
        fp.close()
