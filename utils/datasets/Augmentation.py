import torch
from utils.functionals import crop, super_resolution
import math


class ImageAugmentation:
    def __init__(self, out_shape, stride: int, crop_size: int):
        self.stride = stride
        self.crop_size = crop_size
        self.out_shape = out_shape
        return

    def crop(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        h_crops = math.ceil(h / self.stride)
        w_crops = math.ceil(w / self.stride)

        result = list()
        for i in range(h_crops):
            h_start = min(i * self.stride, h - self.crop_size)
            for j in range(w_crops):
                w_start = min(j * self.stride, w - self.crop_size)
                cropped_image = image[..., h_start: (h_start + self.crop_size), w_start: (w_start + self.crop_size)]
                result.append(cropped_image)

        if len(result) == 0:
            raise ValueError('unable to crop image, check configuration')

        result = torch.concat(result, dim=0)
        return result

    def domain_shift(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        c = image.shape[1]
        image_shifted = list()
        for i in range(c):
            for j in range(i + 1, c):
                img = image.clone()
                img[:, [j, i], ...] = img[:, [i, j], ...] 
                image_shifted.append(img)

        return torch.concat(image_shifted, dim=0)

    def resize(self, image: torch.Tensor):
        if image.shape[-1] < self.out_shape[-1] or image.shape[-2] < self.out_shape[-2]:
            return None

        image = self.crop(image)
        image = torch.nn.functional.interpolate(image, tuple(self.out_shape))
        # image = super_resolution(image, self.out_shape)

        return image

    def __call__(self, image: torch.Tensor) -> dict[torch.Tensor]:
        result = dict()

        # result['cropped'] = self.crop(image)
        cropped_images = self.resize(image)
        result['unshifted'] = cropped_images
        shifted_images = self.domain_shift(cropped_images)
        result['shifted'] = shifted_images

        return result
