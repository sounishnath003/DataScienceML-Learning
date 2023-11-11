FROM python:3.11-slim

WORKDIR /home/twotower/app
RUN pip install --upgrade pip poetry
RUN python -m poetry config virtualenvs.in-project true
COPY pyproject.toml /home/twotower/app/
RUN python -m poetry install
RUN groupadd -r twotower && useradd -r -g twotower -d /home/twotower/app -s /bin/bash twotower
RUN chown -R twotower:twotower /home/twotower/app

USER twotower
COPY . /home/twotower/app/

ENTRYPOINT [ "poetry", "run", "sh", "+x", "run.sh" ]