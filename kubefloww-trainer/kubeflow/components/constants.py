import os
from dataclasses import dataclass


@dataclass
class KubeflowConfiguration:
    BASE_IMAGE: str = "python:3.11-slim"
    ARRTIFACT_DIRECTORY: str = os.path.join(os.getcwd(), "artifacts", "kubeflow_store")
