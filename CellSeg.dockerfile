FROM jupyter/all-spark-notebook:latest as base

ENV DEBIAN_FRONTEND noninteractive
USER root

FROM base as src
WORKDIR /home/jovyan
RUN git clone https://github.com/andrewrech/CellSeg \
      && mkdir -p ./CellSeg/src/modelFiles \
      && cd ./CellSeg/src/modelFiles \
      && wget "https://s3.amazonaws.com/get.rech.io/final_weights.h5"

FROM src as deps
WORKDIR /home/jovyan/CellSeg
COPY . .
RUN bash conda.bash

FROM deps as release
WORKDIR "${HOME}"

ENTRYPOINT ["/usr/bin/bash"]
