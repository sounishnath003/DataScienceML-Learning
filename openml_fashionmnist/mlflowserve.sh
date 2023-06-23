#!/usr/bin/env sh

# Set environment variable for the tracking URL where the Model Registry resides
export MLFLOW_TRACKING_URI=http://localhost:5000
export model_name="openml_fashionmnist_minivgg"
export version="Staging"


# Serve the production model from the model registry
mlflow models serve -m "models:/$model_name/$version" --port 5002