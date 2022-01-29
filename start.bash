#!/usr/bin/bash
# shellcheck shell=bash

cd "$HOME"/CellSeg
conda activate cellsegsegmenter
python3 run.py
