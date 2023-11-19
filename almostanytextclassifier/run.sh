#/bin/bash
clear;
python3 -m poetry run black .
python3 -m poetry run python3 -m almostanytextclassifier.main