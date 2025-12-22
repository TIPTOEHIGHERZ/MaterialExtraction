#!/bin/bash

cwd=$(pwd)

echo "Current Working Directory: $cwd"
python $cwd/scripts/data_process.py --config configs/minc_data_process.yaml --num_workers 32 --device 'cuda:3'
