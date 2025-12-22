#!/bin/bash
# nohup python scripts/train_feature_extracter.py --config_file configs/train_feature_extractor.yaml > output.log 2>&1 &
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_LAUNCH_BLOCKING=1
accelerate launch scripts/train.py --config_file configs/train.yaml
# torchrun --nproc_per_node=4 --master_port=29501 scripts/train.py --config_file configs/train.yaml
# python scripts/train.py --config_file configs/train.yaml
