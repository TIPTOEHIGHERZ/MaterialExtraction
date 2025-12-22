import os
import numpy as np


for i in np.arange(0., 0.5, 0.05):
    os.system(f'python eval_scripts/eval_siamese.py --cond_scale {1. + i}')

# result_dir = './eval/only_render'
# for instance in os.listdir(result_dir):
#     if not instance.endswith('cfg'):
#         continue

#     os.system(f'python eval_scripts/calc_score.py \'{os.path.join(result_dir, instance)}\'')
