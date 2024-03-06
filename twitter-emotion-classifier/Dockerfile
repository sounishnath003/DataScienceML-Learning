FROM python:3.12-slim

EXPOSE 6006
USER root

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/nonroot/app
COPY . /home/nonroot/app

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN ls -al -R

RUN groupadd -r nonroot && useradd -r -g nonroot -d /home/nonroot/app/ -s /bin/bash nonroot
RUN chown -R nonroot:nonroot /home/nonroot/app

USER nonroot
# RUN tensorboard --logdir lightning_logs --bind_all

ENTRYPOINT [ "python3", "-m", "src.main" ]
