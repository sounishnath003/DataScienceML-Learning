
import os
from dataclasses import dataclass

@dataclass
class Configuration:
    KubeflowArtifactStore:str=os.path.join(os.getcwd(), "artifacts", "kubeflow_store")