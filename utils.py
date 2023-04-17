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


def psnr(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return peak_signal_noise_ratio(ground_truth, image, data_range=data_range)


def ssim(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)
