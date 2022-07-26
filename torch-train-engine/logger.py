"""
# _* coding: utf8 *_

filename: logger.py

@author: sounishnath
createdAt: 2022-07-26 16:38:49
"""

import logging


logging.basicConfig(
    # filename="output.log",
    # filemode="w+",
    level=logging.INFO,
    format="%(asctime)s:%(name)s:%(levelname)s:  %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("PyTorch-Model")
