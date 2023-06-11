"""
# _* coding: utf8 *_

filename: template.py

@author: sounishnath
createdAt: 2023-06-11 18:34:37
"""

import logging
import os
import pathlib
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s:",
    handlers=[
        logging.FileHandler("output.log", encoding="utf-8", delay=True),
        logging.StreamHandler(sys.stdout),
    ],
)


def main(project_name: str = os.path.split(os.getcwd())[-1]):
    project_name = "_".join(project_name.split(" "))
    project_setup_files = [
        "./src/main.py",
        f"./src/dataset/dataset.py",
        f"./src/model/model.py",
        f"./src/utils.py",
        "./README.md",
        "./LICENSE.md",
        "./requirements.txt",
        "./webapp/webapp.py",
        ".gitignore",
        "Dockerfile",
        "docker-compose.yaml",
        "./data-ingestion/ingestions/ingest.py",
        "./experiments/notebooks/trials.ipynb",
        "./experiments/codes/codes.py",
        f"./server/server.py",
        f"./server/handlers/handler.py",
    ]

    os.system("python -m pip install --upgrade poetry")
    os.system("python -m poetry init")

    for folderpath in project_setup_files:
        folderpath, filename = os.path.split(pathlib.Path(folderpath))
        if folderpath != "" and not os.path.exists(folderpath):
            logging.info(f"folderpath={folderpath} is not exists....")
            os.makedirs(folderpath, exist_ok=True)
        if (
            not os.path.exists(os.path.join(folderpath, filename))
            or os.path.getsize(os.path.join(folderpath, filename)) == 0
        ):
            logging.info(
                f"file={filename} in folderpath={folderpath} does not exists..."
            )
            with open(os.path.join(folderpath, filename), "w") as file:
                file.write("")
                file.close()


if __name__ == "__main__":
    main(os.pardir)
