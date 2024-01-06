import os
from kfp.dsl import Input, Output, component, Artifact

from .constants import KubeflowConfiguration


@component(
    base_image=KubeflowConfiguration.BASE_IMAGE,
    output_component_file=os.path.join(
        KubeflowConfiguration.ARRTIFACT_DIRECTORY, "components", "generate-dataset.yaml"
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
def generate_mar_file(model_name:str, trained_model_checkpoint_in:Input[Artifact]):
    import argparse
    import subprocess
    """
        model_file = args.model_file
        serialized_file = args.serialized_file
        model_name = args.model_name
        handler = args.handler
        extra_files = args.extra_files
        export_file_path = args.export_path
        requirements_file = args.requirements_file
        config_file = args.config_file
    """

    _BUILD_MAR_CREATION_COMMAND_="""torch-model-archiver \
        --model_file={} \
        --serialized_file={} \
        --model_name={} \
        --handler={} \
        --extra_files={} \
        --export_file_path={} \
        --requirements_file={}
        """.format()
    
    cmd_executor=subprocess.Popen(_BUILD_MAR_CREATION_COMMAND_, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    