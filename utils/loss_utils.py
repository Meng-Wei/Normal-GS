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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import cv2

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

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

# Gaussian Shader
def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out

def predicted_normal_loss(normal, normal_ref, alpha=None, threshold=0.05):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: ( H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()
        weight[weight < threshold] = 0.0
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device)
    else:
        weight = torch.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    n = normal_ref.permute(1,2,0).reshape(-1,3)
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    loss = (w * (1.0 - torch.sum(n * n_pred, axis=-1))).mean()

    return loss

def normal_diff(normal, normal_ref, alpha=None, threshold=0.05):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: ( H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()
        weight[weight < threshold] = 0.0
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device)
    else:
        weight = torch.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()

    if normal_ref.shape[2] == 3:
        n = normal_ref.reshape(-1,3)
    else:
        n = normal_ref.permute(1,2,0).reshape(-1,3)

    if normal.shape[2] == 3:
        n_pred = normal.reshape(-1,3)
    else:
        n_pred = normal.permute(1,2,0).reshape(-1,3)

    mask_num = w.sum(axis=-1)
    cosine_sim = F.cosine_similarity(n, n_pred, dim=-1)
    degDiff = torch.rad2deg(torch.acos(cosine_sim))
    degDiff = (w * degDiff).sum(axis=-1) / mask_num
    return degDiff

# Smoothness
def image_space_gradient(image, axis=0):
    # image: CxHxW
    # axis 0: x_axis; 1: y_axis

    if len(image.shape) == 2:
        if axis == 0:
            return image[:, :-1] - image[:, 1:]
        else:
            return image[:-1, :] - image[1:, :]
    else:
        if axis == 0:
            return image[:, :, :-1] - image[:, :, 1:]
        else:
            return image[:, :-1, :] - image[:, 1:, :]

def bilateral_weighted(diff, weight_diff, sigma=1.0):
    return (diff * torch.exp(-sigma * weight_diff)).mean()

def bilateral_smoothness(value, weight, sigma=1.0, norm=1):
    loss = 0.0
    for axis in (0, 1):
        value_gradient = image_space_gradient(value, axis)
        weight_gradient = image_space_gradient(weight, axis)

        if norm == 1:
            value_gradient = torch.abs(value_gradient)
            weight_gradient = torch.abs(weight_gradient)
        elif norm == 2:
            value_gradient = value_gradient ** 2
            weight_gradient = weight_gradient ** 2
        
        loss += bilateral_weighted(value_gradient, weight_gradient, sigma)
    return loss

def color_gradient_error(render, gt):
    loss = 0.0
    for axis in (0, 1):
        gt_gradient = image_space_gradient(gt, axis).detach()
        render_gradient = image_space_gradient(render, axis)
        loss += l2_loss(render_gradient, gt_gradient)
    return loss

def bilateral_color_gradient_error(render, gt, depth, sigma):
    loss = 0.0
    for axis in (0, 1):
        gt_gradient = image_space_gradient(gt, axis)
        render_gradeint = image_space_gradient(render, axis)
        gradient_diff = (gt_gradient - render_gradeint)**2

        depth_gradient = image_space_gradient(depth, axis).detach()
        depth_g_norm = depth_gradient ** 2
        loss += bilateral_weighted(gradient_diff, depth_g_norm, sigma)
    return loss

def total_variation(value, mask, norm=1):
    loss = 0.0
    for axis in (0, 1):
        value_gradient = image_space_gradient(value, axis)
        if norm == 1:
            loss += torch.abs(value_gradient).mean()
        elif norm == 2:
            loss += (value_gradient ** 2).mean()
    return loss

def cross_entropy_loss(input):
    loss = - input * torch.log(input + 1e-7) - (1 - input) * torch.log(1 - input + 1e-7)
    loss = loss.mean()
    return loss