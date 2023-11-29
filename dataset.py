# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: dataset.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2022/4/9
# -----------------------------------------
import numpy as np
import SimpleITK as sitk
from torch.utils import data


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


class TestData(data.Dataset):
    def __init__(self, theta, L):
        # generate views
        angles = np.linspace(0, 180, theta+1)
        angles = angles[:len(angles) - 1]
        num_angles = len(angles)
        # build parallel rays
        self.rays = []
        for i in range(num_angles):
            self.rays.append(build_coordinate_train(L=L, angle=angles[i]))

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, item):
        ray = self.rays[item]  # (L, L, 2)
        return ray


class TrainData(data.Dataset):
    def __init__(self, theta, sin_path, sample_N):
        self.sample_N = sample_N
        # generate views
        angles = np.linspace(0, 180, theta+1)
        angles = angles[:len(angles) - 1]
        # load sparse-view sinogram
        sin = sitk.GetArrayFromImage(sitk.ReadImage(sin_path))
        num_angles, L = sin.shape
        # store sparse-view sinogram and build parallel rays
        self.rays = []
        self.projections_lines = []
        for i in range(num_angles):
            self.projections_lines.append(sin[i, :])  # (, L)
            self.rays.append(build_coordinate_train(L=L, angle=angles[i]))

        self.projections_lines = np.array(self.projections_lines)
        self.rays = np.array(self.rays)

    def __len__(self):
        return len(self.projections_lines)

    def __getitem__(self, item):
        # sample view
        projection_l = self.projections_lines[item]     # (L, )
        ray = self.rays[item]   # (L, L, 2)
        # sample ray
        sample_indices = np.random.choice(len(projection_l), self.sample_N, replace=False)
        projection_l_sample = projection_l[sample_indices]  # (sample_N)
        ray_sample = ray[sample_indices]    # (sample_N, L, 2)
        return ray_sample, projection_l_sample