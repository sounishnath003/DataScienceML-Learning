#!/bin/bash
# submit training job to Vertex Training with 
# pre-built container using gcloud CLI

export JOB_NAME='openml_fashionmnist_minivgg16'
export REGION='asia-south1'
export IMAGE_URI='asia-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest'
export PACKAGE_PATH='gs://<BUCKET_NAME>/torchf-0.1.0.tar.gz'
export JOB_DIR='gs://<<BUCKKET_NAME>>'

export GOOGLE_APPLICATION_CREDENTIALS='$(pwd)/sounish-cloud-workstation-701639c35724.json'

gcloud beta ai custom-jobs create \
    --display-name=${JOB_NAME} \
    --region ${REGION} \
    --python-package-uris=${PACKAGE_PATH} \
    --worker-pool-spec=replica-count=1,machine-type='n1-standard-8',accelerator-type='NVIDIA_TESLA_T4',accelerator-count=1,executor-image-uri=${IMAGE_URI},python-module='torchf.main' \
    --args="--model-name","openml_fashionmnist_minivgg16","--job-dir",$JOB_DIR \
    --impersonate-service-account="<<SRVC_ACCOUNT>>"
