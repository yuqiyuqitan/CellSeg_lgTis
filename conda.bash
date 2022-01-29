#!/usr/bin/bash
# shellcheck shell=bash

source /opt/conda/etc/profile.d/conda.sh &&
  export PATH="/opt/conda/bin:$PATH"

echo 'source /opt/conda/etc/profile.d/conda.sh &&
  export PATH="/opt/conda/bin:$PATH"' >>"$HOME"/.bashrc

conda create -y --name cellsegsegmenter python=3.6 &&
  conda activate cellsegsegmenter &&
  conda config --add channels conda-forge &&
  conda install -y cytoolz==0.10.0 &&
  pip install -r requirements.txt
