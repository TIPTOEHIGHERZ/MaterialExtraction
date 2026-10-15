import functools, math, random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchvision

from utils.functionals import crop_mid, random_mask
from utils.transforms import RandomThinPlateSpline, remap_normal_axes


def _get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _to_pixel_size(value, reference, default):
    if value is None:
        return default
    if isinstance(value, float) and 0.0 <= value <= 1.0:
        return int(round(reference * value))
    return int(value)


def get_cropsize(theta, d, base_angle=np.pi / 4, eps=1e-6):
    theta0 = theta - base_angle
    theta1 = theta + base_angle
    d0 = d[0] / abs(np.sin(theta0 + eps))
    d1 = d[1] / abs(np.sin(theta1 + eps))
    l = min(d0, d1)
    return int(l * np.sin(base_angle)), int(l * np.cos(base_angle))


def transform(x: torch.Tensor, res=256):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    h, w = x.shape[-2:]
    if h < res or w < res:
        raise
    x = torchvision.transforms.RandomCrop(res)(x)
    if in_dim == 3:
        x = x.squeeze(0)
    return x


def align_attr(color, normal, height, roughness, meta_data: dict, transform_func):
    bsz = color.shape[0]
    thetas = meta_data["theta"]
    concat_attr = [color, normal, height, roughness]
    splits_size = [attr.shape[1] for attr in concat_attr]
    normal_offset = sum(splits_size[:1])
    concat_attr = torch.concat(concat_attr, dim=1)
    angles = [-(thetas[i] / np.pi * 180. + 90.) for i in range(bsz)]
    concat_attr = [torchvision.transforms.functional.rotate(concat_attr[i:i + 1], angles[i]) for i in range(bsz)]
    for i in range(bsz):
        concat_attr[i][:, normal_offset:normal_offset + 3] = remap_normal_axes(
            concat_attr[i][:, normal_offset:normal_offset + 3], angles[i]
        )
    crop_sizes = [get_cropsize(thetas[i], concat_attr[i].shape[-2:]) for i in range(bsz)]
    concat_attr = [crop_mid(concat_attr[i], crop_sizes[i]) for i in range(bsz)]
    concat_attr = torch.concat([transform_func(attr) for attr in concat_attr], dim=0)
    color, normal, height, roughness = concat_attr.split(splits_size, dim=1)
    return color, normal, height, roughness


@dataclass
class RenderAugmentor:
    rotate90_enabled: bool = True
    rotate90_angles: tuple[float, ...] = (0., 90., 180., 270.)
    rotate90_target: str = "ref_and_targets"
    jitter_enabled: bool = True
    jitter_range_deg: float = 10.
    align_rotation_enabled: bool = True
    crop_size: int = 352
    ref_transform_enabled: bool = False
    ref_transform_size: int = 256
    mask_aug_enabled: bool = True
    tps_scale: tuple[float, float] = (0.1, 0.3)
    tps_p: float = 0.8
    occ_size: int = 512
    small_hole_max_tries: int = 4
    large_hole_max_tries: int = 2
    brush_max_tries: int = 8
    small_hole_max_size: int = 256
    large_hole_max_size: int = 512
    brush_min_width: int = 12
    brush_max_width: int = 48
    binarize_thresh: float = 0.5
    photometric_enabled: bool = False

    def __call__(self, image_ref, mask_ref, color, normal, height, roughness, meta_data, device):
        image_ref = image_ref.to(device)
        mask_ref = mask_ref.to(device)[:, :1, ...]
        color = color.to(device)
        normal = normal.to(device)
        height = height.to(device)
        roughness = roughness.to(device)

        if self.rotate90_enabled:
            if tuple(self.rotate90_angles) == (0., 90., 180., 270.):
                ref_flip_angle = random.randint(0, 3) * 90.
            else:
                angle_idx = random.randint(0, len(self.rotate90_angles) - 1)
                ref_flip_angle = float(self.rotate90_angles[angle_idx])
        else:
            ref_flip_angle = 0.
        if self.jitter_enabled:
            rotate_angels = [(random.random() * 2 - 1) * self.jitter_range_deg for _ in range(image_ref.shape[0])]
        else:
            rotate_angels = [0. for _ in range(image_ref.shape[0])]

        # Reference line: image_ref / mask_ref only.
        if self.rotate90_enabled and self.rotate90_target in ("ref_and_targets", "ref_only"):
            image_ref = torchvision.transforms.functional.rotate(image_ref, ref_flip_angle)
            mask_ref = torchvision.transforms.functional.rotate(mask_ref, ref_flip_angle)
        if self.ref_transform_enabled:
            refs = torch.concat([image_ref, mask_ref], dim=1)
            refs = torch.concat(
                [torchvision.transforms.RandomCrop(self.ref_transform_size)(refs[i:i + 1]) for i in range(refs.shape[0])],
                dim=0,
            )
            image_ref, mask_ref = refs.split([image_ref.shape[1], mask_ref.shape[1]], dim=1)

        # Target line: color / normal / height / roughness only.
        if self.rotate90_enabled:
            if self.rotate90_target == "ref_and_targets":
                targets = [color, normal, height, roughness]
                target_splits = [t.shape[1] for t in targets]
                targets = torch.concat(targets, dim=1)
                targets = torchvision.transforms.functional.rotate(targets, ref_flip_angle)
                color, normal, height, roughness = targets.split(target_splits, dim=1)
                normal = remap_normal_axes(normal, ref_flip_angle)
            elif self.rotate90_target == "color_only":
                color = torchvision.transforms.functional.rotate(color, ref_flip_angle)
            elif self.rotate90_target == "ref_only":
                pass
            else:
                raise ValueError(f"unsupported rotate90_target={self.rotate90_target!r}")
        theta = meta_data["theta"]
        meta_data["theta"] = [(t - r / 180 * np.pi) for t, r in zip(theta, rotate_angels)]
        if self.align_rotation_enabled:
            color, normal, height, roughness = align_attr(
                color, normal, height, roughness, meta_data, functools.partial(transform, res=self.crop_size),
            )
        else:
            targets = [color, normal, height, roughness]
            target_splits = [t.shape[1] for t in targets]
            targets = torch.concat(targets, dim=1)
            targets = torch.concat(
                [transform(targets[i:i + 1], res=self.crop_size) for i in range(targets.shape[0])],
                dim=0,
            )
            color, normal, height, roughness = targets.split(target_splits, dim=1)
        if self.mask_aug_enabled:
            scale = random.uniform(self.tps_scale[0], self.tps_scale[1])
            tps_transform = RandomThinPlateSpline(scale=scale, p=self.tps_p)
            augment_mask = torch.concat(
                [
                    torch.tensor(
                        random_mask(
                            self.occ_size,
                            small_hole_max_tries=self.small_hole_max_tries,
                            large_hole_max_tries=self.large_hole_max_tries,
                            small_hole_max_size=self.small_hole_max_size,
                            large_hole_max_size=self.large_hole_max_size,
                            brush_max_tries=self.brush_max_tries,
                            brush_min_width=self.brush_min_width,
                            brush_max_width=self.brush_max_width,
                        ),
                        device=device,
                    ).unsqueeze(0)
                    for _ in range(mask_ref.shape[0])
                ],
                dim=0,
            )
            augment_mask = tps_transform(augment_mask)
            augment_mask = nn.functional.interpolate(augment_mask, mask_ref.shape[-2:])
            mask_ref *= augment_mask
        image_ref = torch.concat(
            [torchvision.transforms.functional.rotate(image_ref[i: i + 1, ...], rotate_angels[i]) for i in range(image_ref.shape[0])]
        )
        mask_ref = torch.concat(
            [torchvision.transforms.functional.rotate(mask_ref[i: i + 1, ...], rotate_angels[i]) for i in range(mask_ref.shape[0])]
        )
        thresh = self.binarize_thresh
        mask_ref[mask_ref > thresh] = 1.
        mask_ref[mask_ref <= thresh] = 0.
        image_ref *= mask_ref
        return image_ref, mask_ref, color, normal, height, roughness


def build_render_augment(augment_config):
    rotate90_config = _get(augment_config, "rotate90", {})
    jitter_config = _get(augment_config, "jitter", {})
    align_config = _get(augment_config, "align_rotation", {})
    crop_config = _get(augment_config, "crop", {})
    ref_transform_config = _get(augment_config, "ref_transform", {})
    mask_config = _get(augment_config, "mask_aug", {})
    photometric_config = _get(augment_config, "photometric", {})

    crop_size = int(_get(crop_config, "size", 352))
    R = int(_get(augment_config, "R", _get(augment_config, "input_resolution", _get(augment_config, "resolution", 512))))
    align_enabled = bool(_get(align_config, "enabled", True))
    bound = math.floor(R / math.sqrt(2)) if align_enabled else R
    if crop_size > bound:
        raise ValueError(f"crop.size exceeds bound: R={R}, bound={bound}, crop.size={crop_size}")

    rotate90_angles = tuple(float(a) for a in _get(rotate90_config, "angles", [0., 90., 180., 270.]))
    jitter_range = _get(jitter_config, "range_deg", 10.)
    if isinstance(jitter_range, (list, tuple)):
        jitter_range = max(abs(float(jitter_range[0])), abs(float(jitter_range[1])))

    tps_scale = _get(mask_config, "tps_scale", [0.1, 0.3])
    if isinstance(tps_scale, (int, float)):
        tps_scale = (float(tps_scale), float(tps_scale))
    else:
        tps_scale = (float(tps_scale[0]), float(tps_scale[1]))

    hole_count_config = _get(mask_config, "hole_count", {})
    hole_size_config = _get(mask_config, "hole_size", {})
    occ_size = int(_get(mask_config, "occ_size", 512))

    return RenderAugmentor(
        rotate90_enabled=bool(_get(rotate90_config, "enabled", True)),
        rotate90_angles=rotate90_angles,
        rotate90_target=str(_get(rotate90_config, "target", "ref_and_targets")),
        jitter_enabled=bool(_get(jitter_config, "enabled", True)),
        jitter_range_deg=float(jitter_range),
        align_rotation_enabled=align_enabled,
        crop_size=crop_size,
        ref_transform_enabled=bool(_get(ref_transform_config, "enabled", False)),
        ref_transform_size=int(_get(ref_transform_config, "size", 256)),
        mask_aug_enabled=bool(_get(mask_config, "enabled", True)),
        tps_scale=tps_scale,
        tps_p=float(_get(mask_config, "tps_p", 0.8)),
        occ_size=occ_size,
        small_hole_max_tries=int(_get(hole_count_config, "small_max_tries", 4)),
        large_hole_max_tries=int(_get(hole_count_config, "large_max_tries", 2)),
        brush_max_tries=int(_get(hole_count_config, "brush_max_tries", 8)),
        small_hole_max_size=_to_pixel_size(_get(hole_size_config, "small_max", None), occ_size, occ_size // 2),
        large_hole_max_size=_to_pixel_size(_get(hole_size_config, "large_max", None), occ_size, occ_size),
        brush_min_width=_to_pixel_size(_get(hole_size_config, "brush_min_width", None), occ_size, 12),
        brush_max_width=_to_pixel_size(_get(hole_size_config, "brush_max_width", None), occ_size, 48),
        binarize_thresh=float(_get(mask_config, "binarize_thresh", 0.5)),
        photometric_enabled=bool(_get(photometric_config, "enabled", False)),
    )
