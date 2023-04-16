# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: utils.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2022/4/9
# -----------------------------------------
import numpy as np
from skimage.metrics import structural_similarity, \
                            peak_signal_noise_ratio


def normalization(data):
    v_max = np.max(data)
    v_min = np.min(data)
    return (data-v_min) / (v_max-v_min)


def build_coordinate_train(L, angle):
    angle_rad = np.deg2rad(angle)
    trans_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ]
    )
    x = np.linspace(-1, 1, L)
    y = np.linspace(-1, 1, L)
    x, y = np.meshgrid(x, y, indexing='ij')  # (L, L), (L, L)
    xy = np.stack([x, y], -1).reshape(-1, 2)  # (L*L, 2)
    xy = xy @ trans_matrix.T  # (L*L, 2) @ (2, 2)
    xy = xy.reshape(L, L, 2)
    return xy


def psnr(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return peak_signal_noise_ratio(ground_truth, image, data_range=data_range)


def ssim(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)
