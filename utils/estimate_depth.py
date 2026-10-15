import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional
from torchvision.transforms import ToPILImage, ToTensor, Resize, Compose
import os
from PIL import Image
import numpy as np
from typing import Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class OmnidataPredictor(nn.Module):
    """
    Class handling the dataset generation and preparation
    """

    def __init__(self,
                 depth_ckpt,
                 normal_ckpt=None):
        super().__init__()

        self.depth_ckpt = depth_ckpt
        self.normal_ckpt = normal_ckpt

        self.normal_predictor = None
        self.normal_data_transform = None
        if self.normal_ckpt is not None:
            self.normal_predictor, self.normal_data_transform = self.get_normal_predictor(self.normal_ckpt)
        self.depth_predictor, self.depth_data_transform = self.get_depth_predictor(self.depth_ckpt)

    def get_normal_predictor(self, ckpt_path):
        from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel

        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)  # DPT Hybrid
        # model.pretrained.model.load_state_dict(torch.load('./models/vit_base_resnet50_384/pytorch_model.bin', weights_only=False, map_location='cpu'))

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        trans_totensor = transforms.Compose([])

        return model, trans_totensor

    def get_depth_predictor(self, ckpt_path):
        from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel

        model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
        # model.pretrained.model.load_state_dict(torch.load('./models/vit_base_resnet50_384/pytorch_model.bin', weights_only=False, map_location='cpu'))

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        trans_totensor = transforms.Compose([transforms.Normalize(mean=0.5, std=0.5)])
        return model, trans_totensor

    def forward(self, x):
        """
        Predicts normal and depth of an image
        :param x:
        :return: Cat of depth and normal
        """
        # Resize to 384x512
        original_shape = x.shape[-2:]
        image = functional.resize(x, [384, 512])

        # Predict the normal and depth
        normal = self.predict_normal(image)
        depth = self.predict_depth(image)
        output = torch.cat([depth, normal], dim=1)

        return functional.resize(output, original_shape)

    def predict_normal(self, image):
        im_tensor = self.normal_data_transform(image).cuda()
        with torch.no_grad():
            prediction = self.normal_predictor(im_tensor).clamp(min=0, max=1)
        # Changing to InteriorVerse convention
        prediction = -(prediction * 2 - 1)
        prediction[:, 0, ...] *= -1
        return prediction

    def predict_depth(self, image):
        parameter = next(self.depth_predictor.parameters())
        image = image.to(device=parameter.device, dtype=parameter.dtype)
        im_tensor = self.depth_data_transform(image)
        with torch.no_grad(), torch.autocast(device_type=parameter.device.type, enabled=False):
            try:
                prediction = self.depth_predictor(im_tensor).clamp(min=0, max=1)
            finally:
                # Omnidata's ViT hooks retain features in a module-global dict.
                pretrained = getattr(self.depth_predictor, "pretrained", None)
                activations = getattr(pretrained, "activations", None)
                if activations is not None:
                    activations.clear()
        return prediction.unsqueeze(1)


class NormalizeRange(torch.nn.Module):
    def __init__(self, output_range: list, input_range: Optional[list] = None, eps=1e-6):
        super().__init__()
        self.output_range = output_range
        self.input_range = input_range
        self.eps = eps

        self.fixed_input_range = input_range is not None

        if self.fixed_input_range:
            self.scale, self.shift = self._get_scale_shift(self.input_range, self.output_range)

    def _get_scale_shift(self, input_range, output_range):
        scale = ((output_range[1] - output_range[0]) /
                 (input_range[1] - input_range[0] + self.eps))
        shift = output_range[0] - input_range[0] * scale
        return scale, shift

    def forward(self, x) -> torch.Tensor:
        """
        Transforms the range of tensor.
        :param x: The input tensor
        :return: The transformed tensor
        """
        if self.fixed_input_range:
            scale, shift = self.scale, self.shift
        else:
            input_range = x.min(), x.max()
            scale, shift = self._get_scale_shift(input_range, self.output_range)
        return x * scale + shift

    def inverse(self, y):
        """
        Inverse transforms the range of tensor.
        :param y: The transformed tensor
        :return: The inverse transformed tensor
        """
        if self.fixed_input_range:
            return (y - self.shift) / self.scale
        else:
            raise NotImplementedError("Inverse transform is not implemented for variable input range")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_range={self.input_range}, output_range={self.output_range})"


class SRGB_2_Linear(object):
    def __call__(self, sample):
        return sample ** 2.2


class Linear_2_SRGB(object):
    def __call__(self, sample):
        return sample ** (1 / 2.2)


def readPNG(filename):
    if not filename:
        raise ValueError("Empty filename")
    image = (np.asarray(Image.open(filename).convert("RGB")) / 255.0).astype(np.float32)
    return image


def load_image(path, linear_space=False):
    try:
        extension = os.path.splitext(path)[1].lower()
        if extension in ['.png', '.jpg', '.jpeg']:
            image = readPNG(path)
            if linear_space:
                image = SRGB_2_Linear()(image)
            return image
        elif extension in ['.exr']:
            raise
            return readEXR(path)
    except Exception:
        print(f"Unable to load {path}")
        raise


class DepthEstimator:
    def __init__(self, pretrained_path, device='cuda'):
        self.model = OmnidataPredictor(
            os.path.join(pretrained_path, 'omnidata_dpt_depth_v2.pth'),
        )
        self.model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

        self.transforms = Compose([Resize(size=[480, 640])])
        self.linear2srgb = Linear_2_SRGB()
        self.device = torch.device(device)

        return
    
    def norm_range(self, image: torch.Tensor, mask=None):
        if mask is None:
            max_val = image.amax(dim=(-1, -2), keepdim=True)
            min_val = image.amin(dim=(-1, -2), keepdim=True)
        else:
            max_val = (image + (1 - mask) * -10000).amax(dim=(-1, -2), keepdim=True)
            min_val = (image + (1 - mask) * 10000).amin(dim=(-1, -2), keepdim=True)

        value_range = (max_val - min_val).abs().clamp_min(torch.finfo(image.dtype).eps)
        return (image - min_val) / value_range

    @torch.no_grad()
    def __call__(self, image: torch.Tensor):
        image = image.to(device=self.device, dtype=torch.float32)
        original_shape = image.shape[-2:]

        with torch.autocast(device_type=self.device.type, enabled=False):
            image = self.transforms(image)
            image = self.linear2srgb(image)

            preds = self.model.predict_depth(image)
            preds = nn.functional.interpolate(preds, original_shape)
            depth = preds[:, 0, ...]
            depth = self.norm_range(depth)
            depth = depth.clip(0., 1.)

        return depth
