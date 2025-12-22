#!/bin/bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=4
# export CUDA_LAUNCH_BLOCKING=1
accelerate launch --main_process_port 29502 scripts/finetune_siamese.py --config_file configs/finetune_siamese.yaml
