import torch
import pickle
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm


with open('./test_files/attn_maps.pkl', 'rb') as file:
    attn_maps = pickle.load(file)

# with open('./test_files/attn_maps_1.pkl', 'wb') as file:
#     attn_maps = pickle.dump(attn_maps[-1:], file)

# with open('./test_files/attn_maps_1.pkl', 'rb') as file:
#     attn_maps = pickle.load(file)

with tqdm.tqdm(attn_maps) as progress_bar:
    for i, attn_map in enumerate(progress_bar):
        # attn_map: torch.Tensor = attn_maps[0]

        # 初始化PCA，降维到2维
        h, l, d = attn_map.shape
        attn_map = attn_map.transpose(0, 1)
        attn_map = attn_map.reshape(l, -1)
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(attn_map)  # 形状：[头数, 2]
        # pca_result = pca_result.view()
        # pca_result = pca_result.transpose(-1, -2)
        pca_result = pca_result.reshape(*[int(math.sqrt(pca_result.shape[0]))] * 2, -1)
        plt.figure(figsize=(6, 6))
        heatmap = plt.imshow(pca_result, cmap='viridis')

        plt.colorbar(heatmap)
        plt.savefig(f'./test_files/pca/pca_{i}.jpg')
        plt.close()
