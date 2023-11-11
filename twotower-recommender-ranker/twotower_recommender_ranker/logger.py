"""
# _* coding: utf8 *_

filename: logger.py

@author: sounishnath
createdAt: 2023-10-28 20:08:44
"""

import logging
import os
import sys


class Logger:
    @staticmethod
    def get_logger():
        log_file_path = os.path.join(os.getcwd(), "activity.log")
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s:TwoTowerRankerRecommenderModel:%(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            handlers=[
                logging.FileHandler(filename=log_file_path),
                logging.StreamHandler(),
            ],
        )
        logger = logging.getLogger("TwoTowerRankerRecommenderModel")
        return logger
