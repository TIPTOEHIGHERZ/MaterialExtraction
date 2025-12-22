#!/bin/bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=6,7
# export CUDA_LAUNCH_BLOCKING=1
accelerate launch --main_process_port 29502 scripts/train_mae.py --config_file configs/train_mae.yaml
