#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt))

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class GrayWorldAssumptionLoss(nn.Module):
    def __init__(self):
        super(GrayWorldAssumptionLoss, self).__init__()

    def forward(self, image):
        mean_r = torch.mean(image[0, :, :])
        mean_g = torch.mean(image[1, :, :])
        mean_b = torch.mean(image[2, :, :])
        mean_gray = (mean_r + mean_g + mean_b) / 3
        loss_r = torch.mean(l2_loss(mean_r, mean_gray))
        loss_g = torch.mean(l2_loss(mean_g, mean_gray))
        loss_b = torch.mean(l2_loss(mean_b, mean_gray))

        loss = loss_r + loss_g + loss_b / 3
        return loss

class Channel_wise_depth_consistency(nn.Module):
    def __init__(self):
        super(Channel_wise_depth_consistency, self).__init__()

    def forward(self, Transmitance_D, beta_d, depth, Transmitance_B, beta_b):
        #print("depth:{}".format(depth))
        #print("Transmitance_B:{}, beta_b:{}".format(Transmitance_B, beta_b))
        Transmitance_D = torch.clamp(Transmitance_D, 0.1, 1.0)
        Transmitance_B = torch.clamp(Transmitance_B, 0.1, 1.0)
        Transmitance_B = 1 - Transmitance_B + 1e-5

        TD = -torch.log(Transmitance_D)
        TB = -torch.log(Transmitance_B)

        depth_D = TD / (beta_d + 1e-5)
        depth_D = (depth_D - depth_D.min())/(depth_D.max() - depth_D.min())
        depth_B = TB / (beta_b + 1e-5)
        depth_B = (depth_B - depth_B.min()) / (depth_B.max() - depth_B.min())
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = depth.repeat(1, 3)

        loss_d = torch.mean(l2_loss(depth_D, depth))
        loss_b = torch.mean(l2_loss(depth_B, depth))

        loss = (loss_d + loss_b) / 2
        return loss


