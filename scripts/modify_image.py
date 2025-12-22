import torch
import sys
import os
from argparse import ArgumentParser
import torch.nn.functional as F

sys.path.append(os.getcwd())
from utils.io import load_image, save_image
from model.GaussianSampler.GaussianSampler import generate_gaussian_kernel


def binary_process(mask: torch.Tensor, threshold=0.1):
    if mask.ndim == 4:
        mask = mask[:, 0, :, :].unsqueeze_(1)
    else:
        mask = mask[0, :, :].unsqueeze_(0)
    idx = mask < threshold
    mask[...] = 1.
    mask[idx] = 0
    return mask


def image2mask(image: torch.Tensor, threshold=1e-3):
    mask = torch.zeros_like(image)
    mask[image > threshold] = 1.
    return mask


def random_fill(image: torch.Tensor, mask: torch.Tensor):
    assert image.ndim == 3
    assert mask.shape[0] == 1

    flattened_image = image.view(image.shape[0], -1)
    flattened_mask = mask.view(mask.shape[0], -1)

    idx_fill = torch.where(flattened_mask < 1e-5)[1]
    idx_sample = torch.where(flattened_mask > 1e-5)[1]
    print(idx_fill)

    pixel_idx = torch.randint(0, len(idx_sample), (len(idx_fill),))
    pixel_idx = idx_sample[pixel_idx]
    pixel_sample = flattened_image[:, pixel_idx]
    flattened_image[:, idx_fill] = pixel_sample
    image_modified = flattened_image.view(*image.shape)

    return image_modified


def generate_background(image: torch.Tensor, mask: torch.Tensor):
    assert image.ndim == 3
    assert mask.shape[0] == 1

    flattened_image = image.view(image.shape[0], -1)
    flattened_mask = mask.view(mask.shape[0], -1)

    idx_sample = torch.where(flattened_mask > 1e-5)[1]

    pixel_idx = torch.randint(0, len(idx_sample), (flattened_image.shape[-1],))
    pixel_idx = idx_sample[pixel_idx]
    pixel_sample = flattened_image[:, pixel_idx]
    background = torch.zeros_like(flattened_image)
    background[:, :] = pixel_sample
    background = background.view(*image.shape)

    return background


def main():
    parser = ArgumentParser('modify_image')
    parser.add_argument('image_path', type=str, help='path for image to be edit')
    parser.add_argument('--mask_path', type=str, default=None, help='path for mask for the image')
    parser.add_argument('--out_path', type=str, help='output path for image to be saved')
    parser.add_argument('--kernel_size', type=int, default=3, help='size of the guassian kernel')
    parser.add_argument('--conv_times', type=int, default=1, help='size of the guassian kernel')


    args = parser.parse_args()
    image_path = args.image_path
    kernel_size = args.kernel_size
    conv_times = args.conv_times
    mask_path = args.mask_path
    if args.out_path is None:
        image_name, ext = os.path.splitext(os.path.basename(image_path))
        out_path = os.path.join(os.path.dirname(image_path),
                                image_name + '_modified' + ext)
    else:
        out_path = args.out_path
    
    image = load_image(image_path)
    # mask = binary_process(load_image(mask_path))
    mask = binary_process(image2mask(image)) if mask_path is None else load_image(mask_path)[0, :, :].unsqueeze(0)
    mask = 1 - mask
    # image_modified = random_fill(image, mask)
    background = generate_background(image, mask)
    image_modified = image * mask + background * (1 - mask)
    gaussian_kernel = generate_gaussian_kernel(kernel_size, 1., channels=1)

    mask = mask.unsqueeze(0)
    mask_gaussian = mask
    # for _ in range(conv_times):
    #     mask_gaussian = F.pad(mask_gaussian, ((kernel_size - 1) // 2, ) * 4)
    #     mask_gaussian = F.conv2d(mask_gaussian, gaussian_kernel)
    #     mask_gaussian = mask_gaussian * mask
    # # mask = F.interpolate(mask, original_size)
    # mask_out_path = os.path.join(os.path.dirname(image_path),
    #                              image_name + '_mask_gaussian.png')
    # mask_gaussian[mask > 1e-3] = mask[mask > 1e-3]
    # mask_gaussian = mask_gaussian.repeat(1, 3, 1, 1)
    image_modified = image_modified * mask_gaussian + background * (1 - mask_gaussian)
    # save_image(mask_gaussian, mask_out_path)
    print(f'saving to {out_path}')
    save_image(image_modified, out_path)
    return


if __name__ == '__main__':
    main()


