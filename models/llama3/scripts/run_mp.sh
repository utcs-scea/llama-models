#!/bin/bash

NGPUS=1
CHECKPOINT_DIR=~/.llama/checkpoints/Llama3.2-11B-Vision

PYTHONPATH=$(git rev-parse --show-toplevel) \
  torchrun --nproc_per_node=$NGPUS \
  -m models.llama3.scripts.mp_completion $CHECKPOINT_DIR \
  --world_size $NGPUS
