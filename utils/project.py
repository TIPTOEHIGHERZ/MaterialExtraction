import numpy as np
import cv2
import torch


def depth_and_rgb_to_colored_point_cloud(depth_map, rgb_image, cx, cy):
    """
    将深度图和 RGB 图像转换为带颜色的 3D 点云。
    
    Args:
        depth_map (np.ndarray): 深度图 (H, W)。
        rgb_image (np.ndarray): RGB 图像 (H, W, 3)。
        fx, fy, cx, cy (float): 相机内参。
        
    Returns:
        points (np.ndarray): 3D 点 (N, 3)，[X, Y, Z]。
        colors (np.ndarray): 颜色 (N, 3)，[R, G, B]。
    """
    H, W = depth_map.shape
    
    # 1. 创建像素坐标网格
    v_grid, u_grid = np.meshgrid(
        np.arange(H), 
        np.arange(W), 
        indexing='ij' # (H, W) 顺序
    )

    # 2. 展平所有数据
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    depth_flat = depth_map.flatten()
    # 将 (H, W, 3) 变为 (H*W, 3)
    channels = rgb_image.shape[-1]
    colors_flat = rgb_image.reshape(-1, channels)

    # 3. 过滤无效深度
    valid_mask = depth_flat > 0
    # print(valid_mask.any())
    
    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]
    depth_valid = depth_flat[valid_mask]
    colors_valid = colors_flat[valid_mask]

    # 4. 反投影到 3D 空间
    X = (u_valid - cx) * depth_valid
    Y = (v_valid - cy) * depth_valid
    Z = depth_valid

    points = np.vstack((X, Y, Z)).T
    return points, colors_valid

def project_point_cloud_to_ortho_image(points, colors, shape):
    """
    将带颜色的点云正交投影 ("splatting") 到一个新的 2D 图像上。
    
    Args:
        points (np.ndarray): 3D 点 (N, 3)，[X, Y, Z]。
        colors (np.ndarray): 颜色 (N, 3)，[R, G, B]。
        meters_per_pixel (float): 新图像中每个像素代表的真实世界米数。
                                 值越小，分辨率越高，但空洞可能越多。
                                 
    Returns:
        ortho_image (np.ndarray): 投影后的新 RGB 图像 (H_new, W_new, 3)。
    """
    
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    
    # 1. 计算 3D 坐标的边界
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)

    # 2. 根据真实世界的米/像素比例，计算新图像的尺寸
    # 我们希望新图像的像素 (0, 0) 对应 (x_min, y_min)
    ratio_x = shape[1] / (x_max - x_min)
    ratio_y = shape[0] / (y_max - y_min)
    W_new = shape[1]
    H_new = shape[0]
    
    # print(f"新图像分辨率 (H, W): ({H_new}, {W_new})")

    # 3. 初始化输出图像和深度缓冲
    # 深度缓冲用于处理遮挡：只有 Z 值更小的点才能被绘制
    channels = colors.shape[-1]
    ortho_image = np.zeros((H_new, W_new, channels), dtype=np.uint8)
    # 用无穷大初始化深度缓冲
    ortho_depth_buffer = np.full((H_new, W_new), np.inf, dtype=np.float32)

    # 4. 排序点云：从远到近绘制 (可选，但 Z-buffer 已处理)
    # (Z-buffer 检查是更稳健的方法，这里我们跳过排序)
    
    # 5. 计算每个点在新图像上的 2D 像素坐标 (u', v')
    u_ortho = ((X - x_min) * ratio_x).astype(int)
    v_ortho = ((Y - y_min) * ratio_y).astype(int)

    # 6. "Splatting" (溅射) 循环
    # 这是性能瓶颈，但在 Numpy 中很难完全向量化
    for i in range(len(points)):
        u, v = u_ortho[i], v_ortho[i]
        
        # 检查是否在图像边界内
        if 0 <= v < H_new and 0 <= u < W_new:
            
            # 检查 Z-buffer (深度缓冲)
            if Z[i] < ortho_depth_buffer[v, u]:
                # 这个点更近，更新像素颜色和深度
                ortho_image[v, u] = colors[i]
                ortho_depth_buffer[v, u] = Z[i]

    all_holes_mask = (ortho_depth_buffer == np.inf).astype(np.uint8)
    fiiled_mask = 1 - all_holes_mask
    
    # 2. 定义一个 3x3 核
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    # 3. 腐蚀 all_holes_mask
    #    isolated_holes_mask: 1 = 该像素及其3x3邻域 *全是* 空洞
    isolated_holes_mask = cv2.erode(all_holes_mask, kernel, iterations=1)
    # isolated_holes_mask = cv2.dilate(all_holes_mask, kernel, iterations=1)
    # isolated_holes_mask = cv2.erode(isolated_holes_mask, kernel, iterations=3)
    # discard_holes_mask = cv2.dilate(1 - all_holes_mask, np.ones((5, 5), dtype=np.uint8), iterations=2)
    
    # 4. 计算智能掩码
    #    smart_inpaint_mask: 1 = 该像素是空洞, *但* 它至少有1个邻居是有数据的
    smart_inpaint_mask = all_holes_mask - isolated_holes_mask
    # smart_inpaint_mask = all_holes_mask

    # 5. 调用 OpenCV Inpaint，*只* 填充边缘空洞
    # image = ortho_image[..., :3]
    image = cv2.inpaint(
        ortho_image[..., :3], 
        smart_inpaint_mask,
        inpaintRadius=3, 
        flags=cv2.INPAINT_TELEA
    )

    if ortho_image.shape[-1] > 3:
        mask = ortho_image[..., 3:]

        mask = cv2.inpaint(
            mask, 
            smart_inpaint_mask,
            inpaintRadius=3, 
            flags=cv2.INPAINT_TELEA
        )
        mask = cv2.erode(mask, kernel, iterations=1)

        return image, mask

    return image


def project(
    image: torch.Tensor | np.ndarray, 
    depth: torch.Tensor | np.ndarray, 
    mask: torch.Tensor | np.ndarray=None,
    shift=0.
):
    image_type = 'np'
    in_dim = 3
    device = 'cpu'

    assert mask is None or (type(mask) == type(image))

    if isinstance(image, torch.Tensor):
        image_type = 'tensor'
        device = image.device
        if image.ndim == 4:
            assert image.shape[0] == 1
            image = image[0, ...]
            depth = depth[0, ...]
            in_dim = 4
        elif image.ndim != 3:
            raise

        image = image.permute(1, 2, 0).cpu().numpy()
        depth = depth.cpu().numpy()

        if mask is not None:
            if mask.ndim == 4:
                assert mask.shape[0] == 1
                mask = mask[0, ...]
            elif mask.ndim != 3:
                raise

            mask = mask.permute(1, 2, 0).cpu().numpy()
            mask[mask >= 0.5] = 1.
            mask[mask < 0.5] = 0.

            depth -= (1 - mask[:, :, 0]) * 10000
            depth += shift

    if image.dtype != np.uint8:
        image = (image * 255.).astype(np.uint8)
    
    if mask.dtype != np.uint8:
        mask = (mask * 255.).astype(np.uint8)

    target_shape = [s // 2 for s in image.shape[:2]]
    # target_shape = [int(s * 1.) for s in image.shape[:2]]

    if mask is not None:
        points, colors_valid = depth_and_rgb_to_colored_point_cloud(depth, np.concatenate([image, mask], axis=-1), image.shape[1] // 2, image.shape[0] // 2)
        ortho_image, mask = project_point_cloud_to_ortho_image(points, colors_valid, shape=target_shape)

        if image_type == 'tensor':
            ortho_image = torch.tensor(ortho_image / 255.).to(device).permute(2, 0, 1)
            mask = torch.tensor(mask / 255.).to(device).permute(2, 0, 1)

            if in_dim == 4:
                ortho_image = ortho_image.unsqueeze(0)
                mask = mask .unsqueeze(0)
        
        return ortho_image, mask
    else:
        points, colors_valid = depth_and_rgb_to_colored_point_cloud(depth, image, image.shape[1] // 2, image.shape[0] // 2)
        ortho_image = project_point_cloud_to_ortho_image(points, colors_valid, shape=target_shape)

        if image_type == 'tensor':
            ortho_image = torch.tensor(ortho_image / 255.).to(device).permute(2, 0, 1)

            if in_dim == 4:
                ortho_image = ortho_image.unsqueeze(0)

        return ortho_image


if __name__ == '__main__':
    import os 
    import sys
    from PIL import Image

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from utils.io import load_image, save_image

    image = load_image('./test_out/image.png')
    depth = load_image('./test_out/depth.png')[0]
    mask = load_image('./test_out/mask.png')

    orth_image, mask = project(image, depth, mask)
    save_image(orth_image, './orth.png')
    save_image(mask, './mask.png')

