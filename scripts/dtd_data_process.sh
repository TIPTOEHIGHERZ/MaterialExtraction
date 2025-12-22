#!/bin/bash

cwd=$(pwd)

echo "Current Working Directory: $cwd"
python $cwd/scripts/data_process.py --config configs/dtd_data_process.yaml
