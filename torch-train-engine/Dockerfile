FROM --platform=linux/amd64 ubuntu:20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y htop python3-dev wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh


RUN conda create -y -n ml python=3.8
COPY . /src

VOLUME [ "/src" ]
RUN /bin/bash -c "cd src \
    && source activate ml \
    && pip install -r requirements.txt \
    && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu"

ENTRYPOINT /bin/bash -c "cd src \
    && source activate ml \
    && python main.py"