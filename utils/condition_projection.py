"""Stage-owned depth projection of condition RGB and its validity mask."""

from pathlib import Path
import math

import torch
from torch.nn import functional as F

from utils.project import project


class ConditionProjector:
    def __init__(self, depth_estimator, *, shift=0.8, depth_batch_size=1):
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("condition projection shift must be finite and positive")
        if not isinstance(depth_batch_size, int) or depth_batch_size < 1:
            raise ValueError("depth_batch_size must be a positive integer")
        self.depth_estimator = depth_estimator
        self.shift = shift
        self.depth_batch_size = depth_batch_size

    @torch.no_grad()
    def __call__(self, image, mask):
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[0] == 0:
            raise ValueError("condition image must have shape [B, 3, H, W], B > 0")
        if (
            mask.ndim != 4
            or mask.shape[1] not in (1, 3)
            or mask.shape[0] != image.shape[0]
            or mask.shape[-2:] != image.shape[-2:]
        ):
            raise ValueError("condition mask must match image with 1 or 3 channels")
        if not image.is_floating_point():
            raise ValueError("condition image must be floating point in [0, 1]")

        images, masks = [], []
        # Depth inference is microbatched; rasterization uses the legacy CPU path.
        for start in range(0, image.shape[0], self.depth_batch_size):
            batch = image[start : start + self.depth_batch_size]
            depth = self.depth_estimator(batch).detach().float().cpu()
            if depth.shape != (batch.shape[0], *batch.shape[-2:]):
                raise ValueError("depth estimator must return [B, H, W]")
            if not torch.isfinite(depth).all():
                raise ValueError("depth estimator returned nonfinite depth")
            batch_cpu = batch.detach().float().cpu()
            mask_cpu = mask[start : start + len(batch), :1].detach().float().cpu()
            for rgb, distance, valid in zip(batch_cpu, depth, mask_cpu):
                projected, projected_mask = project(
                    rgb, distance, valid, shift=self.shift
                )
                images.append(projected)
                masks.append(projected_mask)

        # Match get_projection.py: nearest resize, then a conservative mask cutoff.
        result = F.interpolate(torch.stack(images), image.shape[-2:], mode="nearest")
        valid = F.interpolate(torch.stack(masks), image.shape[-2:], mode="nearest")
        return (
            result.to(device=image.device, dtype=image.dtype),
            (valid >= 0.9).to(device=image.device, dtype=image.dtype),
        )


def build_condition_projector(config, *, device):
    if not config or not config.get("enabled", False):
        return None

    pretrained_path = Path(config.get("pretrained_path", "./pretrained/iid"))
    checkpoint = pretrained_path / "omnidata_dpt_depth_v2.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"condition_projection requires depth weights: {checkpoint}; "
            "set condition_projection.pretrained_path to the Omnidata directory"
        )
    from utils.estimate_depth import DepthEstimator

    try:
        estimator = DepthEstimator(str(pretrained_path), device=device)
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("omnidata_tools"):
            raise ModuleNotFoundError(
                "condition_projection requires the omnidata_tools package "
                "used by utils.estimate_depth"
            ) from error
        raise
    return ConditionProjector(
        estimator,
        shift=config.get("shift", 0.8),
        depth_batch_size=config.get("depth_batch_size", 1),
    )


def prepare_condition(data, pipeline):
    projector = getattr(pipeline, "condition_projector", None)
    has_image = "image_project" in data
    has_mask = "mask_project" in data
    if projector is not None and has_image != has_mask:
        raise ValueError("image_project and mask_project must be supplied together")
    image = data["image_project" if has_image else "image"]
    mask = data["mask_project" if has_mask else "mask"]
    if projector is not None and not has_image:
        return projector(image, mask)
    return image, mask
