import os
import logging
import argparse
from pathlib import Path


def run():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--projectName", type=str, required=True)
    opts, pipeline_opts = parser.parse_known_args()

    project_name = opts.projectName

    logging.info("===================================")
    logging.info("skafolding new project={0}".format(project_name))
    logging.info("===================================")

    project_files_to_be_provisioned = [
        "artifacts/store",
        "setup.py",
        "test.py",
        "logs/activity.log",
        ".gitignore",
        "Dockerfile",
        "trainer/__init__.py",
        "trainer/utils.py",
        "trainer/data_cleanser.py",
        "trainer/dataset.py",
        "trainer/model.py",
        "trainer/lightning.py",
        "main.py",
        "inference.py",
        "config/__init__.py",
        "config/config.py",
        "notebooks/research.ipynb",
        "inference_service/server.py",
        "kubeflow/components/__init__.py",
        "kubeflow/create_pipeline.py",
        "kubeflow/__init__.py",
        ".env",
        "run.sh",
        "requirements.txt",
        "docker-compose.yaml",
        "cloud/cloudbuild.yaml",
        "cloud/workflows.yaml",
    ]

    project_folder = os.path.join(os.getcwd(), project_name)
    if (not os.path.exists(project_folder)) or os.path.getsize(project_folder) == 0:
        os.makedirs(project_folder, exist_ok=True)

    for file in project_files_to_be_provisioned:
        filepath = Path(os.path.join(project_folder, file))
        folder_path = filepath.parent

        if (not os.path.exists(folder_path)) or (os.path.getsize(folder_path) == 0):
            os.makedirs(folder_path, exist_ok=True)

        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w+") as fp:
                fp.write("# write some code...")
                fp.close()
                logging.info("file={0} has been created...".format(filepath))

    logging.info("===================================")
    logging.info("complete setup has been done...")
    logging.info("===================================")


if __name__ == "__main__":
    run()
