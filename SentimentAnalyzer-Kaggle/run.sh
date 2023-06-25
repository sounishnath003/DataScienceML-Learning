#!/usr/bin/bash

clear
cat /dev/null > runs.log
rm -fr lightning_logs
black .
python -m sentiment_kaggle.main