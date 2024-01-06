
FROM pytorch/torchserve:latest-cpu

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY ./torchserve/custom_inference_handler.py /home/model-server/

