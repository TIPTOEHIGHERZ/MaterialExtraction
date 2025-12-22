#!/bin/bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO 
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=4,5
# export CUDA_LAUNCH_BLOCKING=1
accelerate launch --main_process_port 29501 scripts/finetune_controlnet.py --config_file configs/finetune_controlnet.yaml
# accelerate launch --main_process_port 29502 scripts/finetune_render_siamese.py --config_file configs/finetune_render_siamese.yaml
