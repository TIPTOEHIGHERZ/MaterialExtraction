#!/bin/bash
conda activate deep
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_LAUNCH_BLOCKING=1
accelerate launch --main_process_port 29501 scripts/finetune_diffusion.py --config_file configs/finetune_diffusion.yaml
