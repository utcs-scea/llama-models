#!/bin/bash

NGPUS=1
CHECKPOINT_DIR=~/.llama/checkpoints/Llama3.2-11B-Vision

#export CUDA_VISIBLE_DEVICES=0
#echo "Running MPS on Background"
#nvidia-cuda-mps-control -d
#echo set_default_active_thread_percentage 50 | nvidia-cuda-mps-control

PYTHONPATH=$(git rev-parse --show-toplevel) \
  torchrun --nproc_per_node=$NGPUS \
  -m models.llama3.scripts.mp_completion $CHECKPOINT_DIR \
  --world_size $NGPUS


echo "Done running benchmark"
sleep 5
sudo sh -c "echo quit | nvidia-cuda-mps-control"
