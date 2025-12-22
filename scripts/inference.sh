#!/usr/bin/bash
# python scripts/inference.py test_files/Bark007_4K-PNG_Color_unshifted_45_modified.png --mask_path test_files/6_mask.png --config configs/inference.yaml
python scripts/inference.py test_files/6_modified.png test_files/6.jpg --mask_path test_files/6_mask.png --config configs/inference.yaml
