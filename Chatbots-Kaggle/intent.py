"""
# _* coding: utf8 *_

filename: intent.py

@author: sounishnath
createdAt: 2022-12-13 21:18:48
"""

import typing
from dataclasses import dataclass


@dataclass
class IntentModel:
    intent: str
    text: typing.List[str]
    responses: typing.List[str]
