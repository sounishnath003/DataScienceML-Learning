#/bin/bash

clear ;
poetry run black . ; 
poetry run python -m main \
    --datasetFolder=./datasets/ml-100k \
    --n_rows=2300 \
    --modelName=beginner-swan \
    --pretrainedHfModelName=distilbert-base-uncased \
    --train_bs=16 \
    --val_bs=64 \
    --lr=1e-3 \
    --worker=3 \
    --device=auto
