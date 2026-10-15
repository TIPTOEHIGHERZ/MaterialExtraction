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
    valid_mask = (depth_flat > 0) & np.isfinite(depth_flat)
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
    
    H_new, W_new = shape
    channels = colors.shape[-1]
    ortho_image = np.zeros((H_new, W_new, channels), dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if len(points) == 0:
        if channels > 3:
            return ortho_image[..., :3], ortho_image[..., 3:]
        return ortho_image

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    
    # 1. 计算 3D 坐标的边界
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    if x_max == x_min or y_max == y_min:
        if channels > 3:
            return ortho_image[..., :3], ortho_image[..., 3:]
        return ortho_image

    # 2. 根据真实世界的米/像素比例，计算新图像的尺寸
    # 我们希望新图像的像素 (0, 0) 对应 (x_min, y_min)
    ratio_x = shape[1] / (x_max - x_min)
    ratio_y = shape[0] / (y_max - y_min)
    
    # 5. 计算每个点在新图像上的 2D 像素坐标 (u', v')
    u_ortho = ((X - x_min) * ratio_x).astype(int)
    v_ortho = ((Y - y_min) * ratio_y).astype(int)

    # Reduce depth first, then choose the earliest source for exact-depth ties.
    in_bounds = (u_ortho >= 0) & (u_ortho < W_new) & (v_ortho >= 0) & (v_ortho < H_new)
    source_indices = np.flatnonzero(in_bounds)
    pixel_indices = v_ortho[in_bounds] * W_new + u_ortho[in_bounds]
    nearest_depth = np.full(H_new * W_new, np.inf, dtype=np.result_type(Z.dtype, np.float32))
    np.minimum.at(nearest_depth, pixel_indices, Z[in_bounds])
    is_nearest = Z[in_bounds] == nearest_depth[pixel_indices]
    first_source = np.full(H_new * W_new, len(points), dtype=np.intp)
    np.minimum.at(first_source, pixel_indices[is_nearest], source_indices[is_nearest])
    filled = first_source < len(points)
    ortho_image.reshape(-1, channels)[filled] = colors[first_source[filled]]
    all_holes_mask = (~filled).reshape(H_new, W_new).astype(np.uint8)
    
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
        if mask.ndim == 2:
            mask = mask[..., None]

        return image, mask

    return image


def project(
    image: torch.Tensor | np.ndarray, 
    depth: torch.Tensor | np.ndarray, 
    mask: torch.Tensor | np.ndarray=None,
    shift=0.
):
    """Project one CHW/B1CHW tensor or HWC NumPy image without mutating inputs.

    Tensor outputs retain floating dtype/device; NumPy outputs remain uint8.
    Empty or zero-span foregrounds produce an empty image and validity mask.
    """
    is_tensor = isinstance(image, torch.Tensor)
    batched = is_tensor and image.ndim == 4
    if mask is not None and isinstance(mask, torch.Tensor) != is_tensor:
        raise TypeError("image and mask must both be tensors or both be NumPy arrays")

    if is_tensor:
        device = image.device
        image_dtype = image.dtype if image.is_floating_point() else torch.float32
        mask_dtype = mask.dtype if mask is not None and mask.is_floating_point() else image_dtype
        if batched:
            if image.shape[0] != 1:
                raise ValueError("project accepts one image at a time")
            image = image[0]
        if image.ndim != 3:
            raise ValueError("tensor image must have shape CHW or 1CHW")
        image = image.detach().permute(1, 2, 0).cpu()
        image = (image.float() if image.dtype == torch.bfloat16 else image).numpy()
        depth = depth.detach().cpu()
        depth = (depth.float() if depth.dtype == torch.bfloat16 else depth).numpy()
        if mask is not None:
            if mask.ndim == 4 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 3:
                raise ValueError("tensor mask must have shape CHW or 1CHW")
            mask = mask.detach().permute(1, 2, 0).cpu()
            mask = (mask.float() if mask.dtype == torch.bfloat16 else mask).numpy()

    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("image must be RGB and at least 2x2")
    depth = np.asarray(depth)
    while depth.ndim > 2 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.shape != image.shape[:2]:
        raise ValueError("depth and image must have matching spatial dimensions")
    depth = depth.astype(depth.dtype if np.issubdtype(depth.dtype, np.floating) else np.float32, copy=True)
    depth += shift

    if image.dtype != np.uint8:
        image = (image * 255.).astype(np.uint8)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 2:
            mask = mask[..., None]
        if mask.ndim != 3 or mask.shape[:2] != image.shape[:2] or mask.shape[-1] not in (1, 3):
            raise ValueError("mask must match image dimensions and have one or three channels")
        threshold = 127.5 if mask.dtype == np.uint8 else 0.5
        mask = (mask >= threshold).astype(np.uint8) * 255
        depth[mask[..., 0] == 0] = -np.inf

    target_shape = [s // 2 for s in image.shape[:2]]
    colors = image if mask is None else np.concatenate([image, mask], axis=-1)
    points, colors_valid = depth_and_rgb_to_colored_point_cloud(
        depth, colors, image.shape[1] // 2, image.shape[0] // 2
    )
    projected = project_point_cloud_to_ortho_image(points, colors_valid, shape=target_shape)
    if not is_tensor:
        return projected

    arrays = (projected,) if mask is None else projected
    outputs = []
    for array, dtype in zip(arrays, (image_dtype, mask_dtype)):
        output = torch.from_numpy(array).to(device=device, dtype=dtype).permute(2, 0, 1) / 255.
        outputs.append(output.unsqueeze(0) if batched else output)
    return outputs[0] if mask is None else tuple(outputs)
