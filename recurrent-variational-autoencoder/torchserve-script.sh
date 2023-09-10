#!/usr/bin/bash

poetry run torchserve --stop
poetry run black .
clear
poetry run torch-model-archiver \
    --model-name=rvae-neuralnet \
    --version=1.0 \
    --model-file=./recurrent_variation_autoencoder/model.py \
    --serialized-file=./lightning_logs/version_0/checkpoints/epoch=1-step=20.ckpt \
    --handler rvae_endpoint.py \

rm -fr model_store logs
mkdir model_store
mv rvae-neuralnet.mar model_store

poetry run torchserve --start \
    --model-store model_store \
    --models rvae-neuralnet.mar \
    --ncs


curl -X POST http://localhost:8080/predictions/rvae-neuralnet \
   -H "Content-Type: application/json" \
   -d '{"productId": 123456, "quantity": 100}'  