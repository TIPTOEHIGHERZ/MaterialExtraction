import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms.functional as TF

from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.core import Tensor, tensor
from kornia.geometry.transform import get_tps_transform, warp_image_tps


__all__ = ['RandomThinPlateSpline', 'remap_normal_axes', 'rotate_normal']


def _to_radians_batch(
    angle: Union[float, int, Sequence, torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize a scalar / per-sample angle (in degrees) to a (B,) tensor of radians."""
    if isinstance(angle, torch.Tensor):
        a = angle
        if a.ndim == 0:
            a = a.reshape(1).expand(batch_size)
        else:
            assert a.shape[0] == batch_size, f'angle batch {a.shape[0]} != normal batch {batch_size}'
        a = a.to(device=device, dtype=dtype)
    elif isinstance(angle, (int, float)):
        a = torch.full((batch_size,), float(angle), device=device, dtype=dtype)
    else:
        seq = list(angle)
        assert len(seq) == batch_size, f'angle batch {len(seq)} != normal batch {batch_size}'
        a = torch.tensor([float(x) for x in seq], device=device, dtype=dtype)
    return a * (math.pi / 180.0)


def remap_normal_axes(
    normal: torch.Tensor,
    angle: Union[float, int, Sequence, torch.Tensor],
    encoded: bool = True,
    convention: str = 'opengl',
) -> torch.Tensor:
    """In-plane rotate the (nx, ny) channels of an OpenGL normal map.

    Pixel positions are unchanged. Call this AFTER you have already rotated the spatial
    layout of a tensor that contains a normal slice with `torchvision.transforms.functional.rotate`
    (or any equivalent), passing the SAME angle, so the tangent-space frame tracks the new
    image frame.

    Args:
        normal: `[B, 3, H, W]` or `[3, H, W]`; channels are (nx, ny, nz).
        angle: scalar or length-B sequence/Tensor, degrees, CCW (matches
            `torchvision.transforms.functional.rotate`).
        encoded: `True` (default) when the input lives in `[0, 1]` (the storage form used
            by `load_image`); decoded to `[-1, 1]` internally and re-encoded on the way out.
            Pass `False` when the input is already in `[-1, 1]`.
        convention: `'opengl'` (default; `+x` right, `+y` up). `'directx'` reserved.

    Returns:
        A tensor of the same shape and dtype as `normal`.
    """
    if convention != 'opengl':
        raise NotImplementedError(f'convention={convention!r} is not implemented')
    squeezed = normal.ndim == 3
    if squeezed:
        normal = normal.unsqueeze(0)
    assert normal.ndim == 4 and normal.shape[1] == 3, (
        f'expected normal of shape [B,3,H,W] or [3,H,W], got {tuple(normal.shape)}'
    )
    n = normal * 2.0 - 1.0 if encoded else normal
    theta = _to_radians_batch(angle, n.shape[0], n.device, n.dtype)
    cos = torch.cos(theta).reshape(-1, 1, 1)
    sin = torch.sin(theta).reshape(-1, 1, 1)
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    new_nx = nx * cos - ny * sin
    new_ny = nx * sin + ny * cos
    out = torch.stack([new_nx, new_ny, nz], dim=1)
    if encoded:
        out = (out + 1.0) / 2.0
    if squeezed:
        out = out.squeeze(0)
    return out


def rotate_normal(
    normal: torch.Tensor,
    angle: Union[float, int, Sequence, torch.Tensor],
    encoded: bool = True,
    convention: str = 'opengl',
    interpolation: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
    expand: bool = False,
    fill: Optional[list] = None,
) -> torch.Tensor:
    """Spatial-rotate a normal map AND remap its (nx, ny) channels accordingly.

    Equivalent to:
        rotated = torchvision.transforms.functional.rotate(normal, angle)
        result  = remap_normal_axes(rotated, angle, encoded=..., convention=...)
    Use this when the caller owns a pure-normal tensor. For packed tensors (e.g.
    `cat([color, normal, height, roughness], dim=1)`), rotate the pack with torchvision
    and then call `remap_normal_axes` on the normal slice only — that avoids interpolating
    the other channels twice.
    """
    if convention != 'opengl':
        raise NotImplementedError(f'convention={convention!r} is not implemented')
    if isinstance(angle, (int, float)):
        rotated = TF.rotate(normal, float(angle), interpolation=interpolation, expand=expand, fill=fill)
        return remap_normal_axes(rotated, angle, encoded=encoded, convention=convention)
    squeezed = normal.ndim == 3
    if squeezed:
        normal = normal.unsqueeze(0)
    angles = (
        [float(a.item()) for a in angle.reshape(-1)]
        if isinstance(angle, torch.Tensor) else
        [float(a) for a in angle]
    )
    assert len(angles) == normal.shape[0], f'angle len {len(angles)} != batch {normal.shape[0]}'
    parts = [
        TF.rotate(normal[i:i + 1], a, interpolation=interpolation, expand=expand, fill=fill)
        for i, a in enumerate(angles)
    ]
    rotated = torch.cat(parts, dim=0)
    out = remap_normal_axes(rotated, angles, encoded=encoded, convention=convention)
    return out.squeeze(0) if squeezed else out


# NOTE: This NEEDS to be updated. It is out of the random generator controller.
class RandomThinPlateSpline(AugmentationBase2D):
    r"""Add random noise to the Thin Plate Spline algorithm.

    .. image:: _static/img/RandomThinPlateSpline.png

    Args:
        scale: the scale factor to apply to the destination points.
        align_corners: Interpolation flag used by ``grid_sample``.
        mode: Interpolation mode used by `grid_sample`. Either 'bilinear' or 'nearest'.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_image_tps`.

    Examples:
        >>> img = torch.ones(1, 1, 2, 2)
        >>> out = RandomThinPlateSpline()(img)
        >>> out.shape
        torch.Size([1, 1, 2, 2])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomThinPlateSpline(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        scale: float = 0.2,
        align_corners: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        batch_size = None
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.flags = {"align_corners": align_corners}
        self.dist = torch.distributions.Uniform(-scale, scale)
        self.batch_size = batch_size

    def generate_parameters(self, shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        B, _, _, _ = shape
        src = tensor([[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]]).expand(B, 5, 2)  # Bx5x2
        dst = src + self.dist.rsample(src.shape)
        return {"src": src, "dst": dst}

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        if self.same_on_batch:
            batch_size = input.shape[0]
            if self.batch_size is not None:
                batch_size = self.batch_size
            
            src = params["src"][:input.shape[0] // batch_size].to(input)
            src = src.repeat(input.shape[0] // src.shape[0], *([1] * (src.ndim - 1)))
            dst = params["dst"][:input.shape[0] // batch_size].to(input)
            dst = dst.repeat(input.shape[0] // dst.shape[0], *([1] * (dst.ndim - 1)))

        else:
            src = params["src"].to(input)
            dst = params["dst"].to(input)
        # NOTE: warp_image_tps need to use inverse parameters
        kernel, affine = get_tps_transform(dst, src)
        return warp_image_tps(input, src, kernel, affine, flags["align_corners"])
