import torch
import numpy as np
from PIL import Image
from typing import Union


def gaussian_pdf(point: torch.Tensor, mean, std, device='cpu'):
    assert len(mean) == len(std) and len(mean) == point.shape[-1]
    dims = len(mean)
    mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, device=device)
    std = std if isinstance(std, torch.Tensor) else torch.tensor(std, device=device)
    mean = mean.reshape(1, dims)
    std = std.reshape(1, dims)
    exponent_coefficient = (point - mean) / std
    exponent_coefficient = torch.sum(torch.pow(exponent_coefficient, 2.), dim=-1) * -0.5

    return torch.exp(exponent_coefficient)


def apply_gaussian(image: torch.Tensor, mean, std):
    assert len(mean) == len(std)
    device = image.device
    h, w = image.shape[-2:]
    xx, yy = torch.meshgrid(torch.arange(0, w, device=device), torch.arange(0, h, device=device), indexing='ij')
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    points = torch.concat([xx, yy], dim=-1)
    values = gaussian_pdf(points, mean, std, device=device)
    image = image.permute(-2, -1, *range(image.ndim - 2))
    image[points[:, 0], points[:, 1]] = values.reshape(-1, 1)

    return image.permute(*list(reversed(range(image.ndim - 1, 1, -1))), 0, 1)


def generate_gaussian_kernel(kernel_size: int, std, channels=3, device='cpu'):
    xx, yy = torch.meshgrid(torch.arange(0, kernel_size, device=device), 
                            torch.arange(0, kernel_size, device=device),
                            indexing='ij')
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    points = torch.concat([xx, yy], dim=-1)
    mean = [(kernel_size - 1) // 2] * 2
    guassian_kernel = gaussian_pdf(points, mean, [std, std], device=device).reshape(kernel_size, kernel_size)
    guassian_kernel = guassian_kernel / torch.sum(guassian_kernel)
    guassian_kernel = guassian_kernel.unsqueeze(0).unsqueeze(1).repeat(1, channels, 1, 1)

    return guassian_kernel


def rand_point(lows, highs, n_points, device='cpu'):
    point = list()

    for i in range(len(lows)):
        point.append(torch.randint(lows[i], highs[i], [n_points, 1], device=device))

    point = torch.concat(point, dim=-1)
    return list(point)


class GaussianSampler:
    def __init__(self, image_shape, segments, channels=3, device='cpu'):
        """
        Args:
            image_shape: shape for input image

            segments: blocks to devide an image
        """
        self.segments = segments
        self.image_shape = image_shape
        self.channels = channels
        self.xx = torch.linspace(0, image_shape[0], segments[0], device=device).int()
        self.yy = torch.linspace(0, image_shape[1], segments[1], device=device).int()
        self.device = device

        return

    def sample_points(self, n_points=1, device='cpu'):
        points = list()
        for i in range(len(self.xx) - 1):
            for j in range(len(self.yy) - 1):
                points += rand_point(lows=(self.xx[i], self.yy[j]), highs=(self.xx[i + 1], self.yy[j + 1]),
                                     n_points=n_points, device=device)

        return points

    def generate_gaussian(self, n_points, std, gt_radius=None, device=None, to_batch=False):
        device = self.device if device is None else device
        points = self.sample_points(n_points, device=device)
        std = std if isinstance(std, torch.Tensor) else torch.tensor(std, device=device)
        mask = torch.zeros(self.channels, *self.image_shape, device=device)
        for point in points:
            mask += apply_gaussian(torch.zeros_like(mask, device=device), mean=point, std=std)

        if gt_radius is not None:
            gt_radius = gt_radius if isinstance(gt_radius, torch.Tensor) else torch.tensor([gt_radius], device=device)
            gt_radius = torch.min(torch.cat([gt_radius, std]))
            radius_coefficient = torch.exp((gt_radius ** 2) / 2)
            mask = mask * radius_coefficient
        
        if to_batch:
            mask = mask.unsqueeze(0)

        return torch.clamp(mask, max=1.)


if __name__ == '__main__':
    gaussian_sampler = GaussianSampler((512, 512), (8, 8))
    mask = gaussian_sampler.generate_gaussian(1, (10., 10.), gt_radius=0.)
    mask = mask.permute(1, 2, 0)
    mask = mask.cpu().numpy() * 255
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.show('1')
