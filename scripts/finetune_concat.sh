#!/bin/bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=6,7
# export CUDA_LAUNCH_BLOCKING=1
accelerate launch --main_process_port 29501 scripts/finetune_concat.py --config_file configs/finetune_concat.yaml
# accelerate launch --main_process_port 29502 scripts/finetune_render_siamese.py --config_file configs/finetune_render_siamese.yaml
