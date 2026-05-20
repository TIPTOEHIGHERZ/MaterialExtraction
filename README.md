# MatE: Material Extraction from Single-Image via Geometric Prior

This is the official repository for **MatE: Material Extraction from Single-Image via Geometric Prior**.

Paper: [arXiv:2512.18312](https://arxiv.org/abs/2512.18312)

## Status

We have released the evaluation dataset first. The training code, inference/evaluation scripts, and pretrained model weights are currently being organized and will be released in a subsequent update.

## Dataset

The Poly Haven material extraction evaluation set is available on Hugging Face:

[https://huggingface.co/datasets/tiptoez/MatE_polyhaven_extraction](https://huggingface.co/datasets/tiptoez/MatE_polyhaven_extraction)

This dataset is used to evaluate material extraction from a single input image and a user-provided mask. Each sample contains:

- `image.png`: rendered/input image
- `mask.png`: binary mask
- `albedo.png`: ground-truth diffuse/albedo map
- `normal_gl.png`: OpenGL normal map
- `roughness.png`: roughness map
- `height.png`: displacement/height map

The dataset files were normalized only for consistent browsing, downloading, and loading. The normalization standardizes directory names, file names, image modes, and bit depth; it is not intended to change the evaluation target or introduce additional semantic processing.

## Release Plan

- Evaluation dataset: released
- Training code: coming soon
- Inference and evaluation scripts: coming soon
- Pretrained model weights: coming soon

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2025mate,
  title={MatE: Material Extraction from Single-Image via Geometric Prior},
  author={Zhang, Zeyu and Zhai, Wei and Yang, Jian and Cao, Yang},
  journal={arXiv preprint arXiv:2512.18312},
  year={2025}
}
```
