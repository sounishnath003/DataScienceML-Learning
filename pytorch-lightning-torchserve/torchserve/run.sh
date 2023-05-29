#!/bin/bash

torchserve --stop

rm -fr ./**/*.mar ./**/model_store logs
mkdir ./torchserve/model_store

torch-model-archiver \
--model-name imdb-clf-model \
--version 1.0 \
--model-file ./torchh/model.py \
--serialized-file ./models/best_model.ckpt \
--handler ./torchserve/handler.py \
--extra-files ./torchh/dataset.py \
-r ./requirements.txt

mv ./imdb-clf-model.mar ./torchserve/model_store

torchserve \
--start --ncs \
--model-store=./torchserve/model_store \
--models=./torchserve/model_store/imdb-clf-model.mar