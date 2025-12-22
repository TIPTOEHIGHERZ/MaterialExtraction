import os
import numpy as np
import omegaconf


data_path = './datasets/render_result_matsynth_10_resized'

files = [file for file in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, file))]
train_idx = np.random.choice(len(files), 5000, replace=False)

train_files = list()
test_files = list()

for i, file in enumerate(files):
    if i in train_idx:
        train_files.append(file)
    else:
        test_files.append(file)

omegaconf.OmegaConf.save({
    'train': train_files,
    'test': test_files
}, os.path.join(data_path, 'seperate.yaml'))
print(os.path.join(data_path, 'seperate.yaml'))
