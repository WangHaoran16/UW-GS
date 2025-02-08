import torch
import torch.nn.functional as F


def robust_mask(error_pixel, inlier_threshold, inlier_quantile=0.9, smooth_window_size=3, smooth_threshold=0.5,
                outlier_threshold=0.5):

    epsilon = 0.001
    error_pixel = error_pixel.unsqueeze(0)
    error_pixel = torch.permute(error_pixel, [0, 2, 3, 1])
    error_pixel = torch.mean(error_pixel, dim=-1, keepdim=True)

    next_inlier_threshold = torch.quantile(error_pixel, inlier_quantile)
    stats = {
        'inlier_threshold': next_inlier_threshold,
    }

    is_inlier_loss = (error_pixel < inlier_threshold)
    is_inlier_loss = torch.tensor(is_inlier_loss).float()

    stats['is_inlier_loss'] = torch.mean(is_inlier_loss)

    window = torch.ones((1, 1, smooth_window_size, smooth_window_size)) / (smooth_window_size * smooth_window_size)
    window = window.cuda()
    is_inlier_loss_transposed = torch.permute(is_inlier_loss, [0, 3, 1, 2]).cuda()
    has_inlier_neighbors = torch.nn.functional.conv2d(is_inlier_loss_transposed, window, padding=1)
    has_inlier_neighbors = torch.permute(has_inlier_neighbors, [0, 2, 3, 1])
    # Binarize after smoothing.
    has_inlier_neighbors = (
            has_inlier_neighbors > 1 - smooth_threshold
    )
    has_inlier_neighbors = torch.tensor(has_inlier_neighbors).float()
    stats['has_inlier_neighbors'] = torch.mean(has_inlier_neighbors)

    slice_window_size = 16
    slice_window = torch.ones((1, 1, slice_window_size, slice_window_size)) / (slice_window_size * slice_window_size)
    slice_window = slice_window.cuda()

    patch_inlier_indicator = torch.conv2d(is_inlier_loss_transposed, slice_window, stride=8)
    patch_inlier_indicator = (patch_inlier_indicator > 1 - outlier_threshold)
    patch_inlier_indicator = torch.tensor(patch_inlier_indicator).float()
    is_inlier_patch = F.interpolate(patch_inlier_indicator, scale_factor=(8, 8), mode="nearest")
    is_inlier_patch = torch.permute(is_inlier_patch, [0, 2, 3, 1])
    is_inlier_patch = _robust_padding(is_inlier_patch, has_inlier_neighbors, slice_window_size)

    mask = (is_inlier_patch + has_inlier_neighbors + is_inlier_loss > epsilon)
    mask_1 = ( is_inlier_loss > epsilon)
    mask_2 = ( has_inlier_neighbors + is_inlier_loss > epsilon)
    mask = torch.tensor(mask).float()
    mask = torch.permute(mask, [0, 3, 1, 2])
    mask = mask.squeeze(0)
    mask_1 = torch.tensor(mask_1).float()
    mask_1 = torch.permute(mask_1, [0, 3, 1, 2])
    mask_1 = mask_1.squeeze(0)
    mask_2 = torch.tensor(mask_2).float()
    mask_2 = torch.permute(mask_2, [0, 3, 1, 2])
    mask_2 = mask_2.squeeze(0)
    stats['mask'] = torch.mean(mask)

    return mask, stats, mask_1, mask_2


def _robust_padding(up, original,window_size, kernal_seize=8):
    height = original.size(1)
    up_height = up.size(1)
    width = original.size(2)
    up_width = up.size(2)
    lefttop_pad = (window_size-kernal_seize)//2
    pad_size_height = height - up_height
    pad_size_width = width - up_width
    pad_size_lower = 0
    is_inlier_patch = F.pad(up,
                 (0, 0, lefttop_pad, pad_size_width-lefttop_pad, lefttop_pad, pad_size_height-lefttop_pad, 0, 0))
    return is_inlier_patch


