import os
from dataclasses import dataclass


@dataclass
class KubeflowConfig:
    root_directory: str = os.path.join(os.getcwd(), "artifacts")
    base_image: str = "python:3.11-slim"
    packages_to_install = ["pyarrow", "pandas", "scikit-learn", "lightning"]
    save_to = lambda filename: os.path.join(KubeflowConfig.root_directory, filename)
