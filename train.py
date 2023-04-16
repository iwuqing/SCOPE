# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: train_COVID19.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2022/4/9
# -----------------------------------------
import SimpleITK as sitk
import numpy as np
import torch
import datetime
import dataset
import tinycudann as tcnn
import commentjson as json
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

if __name__ == '__main__':

    output_path = './output/sino'
    # load config
    with open('config.json') as config_file:
        config = json.load(config_file)

    # para
    # -----------------------
    in_view = 90
    sample_id = 80
    sin_path = 'data/{}_sino_0_{}.nii'.format(in_view, sample_id)
    lr = 1e-3
    epoch = 5000
    gpu = 0
    summary_epoch = 500
    sample_N = 8
    batch_size = 3
    scale = int(720 / in_view)  # 720/theta
    # the size of each profile in sinogram.
    L = int(np.max(sitk.GetArrayFromImage(sitk.ReadImage(sin_path)).shape))
    projection_angle_num = 720

    # data
    # -----------------------
    train_loader = data.DataLoader(
        dataset=dataset.TrainData(sin_path=sin_path, theta=in_view, sample_N=sample_N),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = data.DataLoader(
        dataset=dataset.TestData(theta=projection_angle_num, L=L),
        batch_size=3,
        shuffle=False
    )

    # model & optimizer
    # -----------------------
    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))

    l1_loss_function = torch.nn.L1Loss()  # L1 Loss
    SCOPE = tcnn.NetworkWithInputEncoding(n_input_dims=2,
                                          n_output_dims=1,
                                          encoding_config=config["encoding"],
                                          network_config=config["network"]).to(DEVICE)
    optimizer = torch.optim.Adam(params=SCOPE.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    for e in range(epoch):
        SCOPE.train()
        loss_train = 0
        for i, (ray_sample, projection_l_sample) in enumerate(train_loader):
            # the sampled rays and the corresponding projections
            ray_sample = ray_sample.to(DEVICE).float().view(-1, 2)  # (N, sample_N, L, 2)
            projection_l_sample = projection_l_sample.to(DEVICE).float()  # (N, sample_N)
            # forward
            pre_intensity = SCOPE(ray_sample).view(batch_size, sample_N, L, 1)  # (N, sample_N, L, 1)
            projection_l_sample_pre = torch.sum(pre_intensity, dim=2)  # (N, sample_N, 1, 1)
            # reshape
            projection_l_sample_pre = projection_l_sample_pre.squeeze(-1).squeeze(-1)  # (N, sample_N)
            # compute loss
            mse_loss = l1_loss_function(projection_l_sample_pre,
                                        projection_l_sample.to(projection_l_sample_pre.dtype))
            loss = mse_loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record and print loss
            loss_train += loss.item()
            print('{}_{}, (TRAIN0) Epoch[{}/{}], Steps[{}/{}], Lr:{}, Loss:{:.6f}'.
                  format(sample_id, in_view, e + 1, epoch, i + 1, len(train_loader), scheduler.get_last_lr()[0],
                         loss.item()))
        scheduler.step()

        if (e + 1) % summary_epoch == 0:
            sin_pre = np.zeros(shape=(projection_angle_num, L))
            with torch.no_grad():
                SCOPE.eval()
                for i, (ray_sample) in enumerate(test_loader):
                    print(i, len(test_loader))
                    # all the parallel rays from each view
                    ray_sample = ray_sample.to(DEVICE).float().view(-1, 2)  # (N, L, L, 2)
                    # forward
                    pre_intensity = SCOPE(ray_sample).view(-1, L, L, 1)  # (N, L, L, 1)
                    # projection i.e, Equ. 2
                    projection_l_sample_pre = torch.sum(pre_intensity, dim=2)  # (N, L, 1, 1)
                    # reshape and store
                    projection_l_sample_pre = projection_l_sample_pre.squeeze(-1).squeeze(-1)  # (N, L)
                    temp = projection_l_sample_pre.cpu().detach().float().numpy()
                    if i == 0:
                        sin_pre = temp
                    else:
                        sin_pre = np.concatenate((sin_pre, temp), axis=0)

            # data consistency
            sin_original = sitk.GetArrayFromImage(sitk.ReadImage(sin_path))
            k = 0
            for i in range(len(sin_pre)):
                if i % scale == 0:
                    sin_pre[i, :] = sin_original[k, :]
                    k = k + 1
            # write dense-view sinogram and model
            sin_pre = sitk.GetImageFromArray(sin_pre)
            sitk.WriteImage(sin_pre, '{}/sin_pre_{}_{}.nii'.format(output_path, sample_id, in_view))
            torch.save(SCOPE.state_dict(), 'model/model_param_covid19_{}_{}.pkl'.format(sample_id, in_view))