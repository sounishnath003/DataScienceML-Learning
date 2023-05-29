#!/bin/bash

TOKENIZERS_PARALLELISM=false
black .
clear
python3 -m src.main