#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
# LPAI owns transport settings; skip the shared trainer's NCCL defaults.
export TEX_EDIT_SET_NCCL_ENV=0

unset LD_LIBRARY_PATH
source ~/envs/blender/bin/activate

exec accelerate launch \
  --multi_gpu \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision no \
  --main_process_port "${MAIN_PROCESS_PORT:-29501}" \
  scripts/train.py \
  --config_file configs/finetune_render_pretrained_two_stage.yaml
