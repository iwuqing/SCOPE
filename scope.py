# ----------------------------------------------#
# Pro    : SCOPE
# File   : reprojection.py
# Date   : 2023/4/17
# Author : Qing Wu                        
# Email  : wuqing@shanghaitech.edu.cn        
# ----------------------------------------------#
import SimpleITK as sitk
import numpy as np
import torch
import dataset
import tinycudann as tcnn
import commentjson as json
from torch.utils import data
from torch.optim import lr_scheduler


def train(config_path):

    # config
    # ------------------------------------
    with open(config_path) as config_file:
        config = json.load(config_file)

    sv_sino_in_path = config["file"]["sv_sino_in_path"]
    dv_sino_out_path = config["file"]["dv_sino_out_path"]
    model_path = config["file"]["model_path"]
    num_sv, num_dv, L = config["file"]["num_sv"], config["file"]["num_dv"], config["file"]["L"]

    lr = config["train"]["lr"]
    epoch = config["train"]["epoch"]
    gpu = config["train"]["gpu"]
    summary_epoch = config["train"]["summary_epoch"]
    sample_N = config["train"]["sample_N"]
    batch_size = config["train"]["batch_size"]

    # data
    # ------------------------------------
    train_loader = data.DataLoader(
        dataset=dataset.TrainData(sin_path=sv_sino_in_path, theta=num_sv, sample_N=sample_N),
        batch_size=batch_size,
        shuffle=True
    )

    # model & optimizer
    # ------------------------------------
    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))

    l1_loss_function = torch.nn.L1Loss()  # L1 Loss
    SCOPE = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                          encoding_config=config["encoding"],
                                          network_config=config["network"]).to(DEVICE)
    optimizer = torch.optim.Adam(params=SCOPE.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    # train
    # ------------------------------------
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
            loss = l1_loss_function(projection_l_sample_pre, 
                                    projection_l_sample.to(projection_l_sample_pre.dtype))
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record and print loss
            loss_train += loss.item()
        scheduler.step()
        print('{}, (TRAIN0) Epoch[{}/{}], Lr:{}, Loss:{:.6f}'.
              format(num_sv, e + 1, epoch, scheduler.get_last_lr()[0], loss_train/len(train_loader)))

        if (e + 1) % summary_epoch == 0:
            torch.save(SCOPE.state_dict(), model_path)



def reprojection(config_path):

    # config
    # ------------------------------------
    with open(config_path) as config_file:
        config = json.load(config_file)

    sv_sino_in_path = config["file"]["sv_sino_in_path"]
    dv_sino_out_path = config["file"]["dv_sino_out_path"]
    model_path = config["file"]["model_path"]
    num_sv, num_dv, L = config["file"]["num_sv"], config["file"]["num_dv"], config["file"]["L"]
    scale = int(num_dv/num_sv)
    batch_size = config["train"]["batch_size"]
    gpu = config["train"]["gpu"]
    
    # dataloader
    # ------------------------------------
    test_loader = data.DataLoader(
        dataset=dataset.TestData(theta=num_dv, L=L),
        batch_size=batch_size,
        shuffle=False
    )

    # model
    # ------------------------------------
    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))
    SCOPE = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                          encoding_config=config["encoding"],
                                          network_config=config["network"]).to(DEVICE)
    SCOPE.load_state_dict(torch.load(model_path))


    # reprojetion
    # ------------------------------------
    sin_pre = np.zeros(shape=(num_dv, L))
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
    sin_original = sitk.GetArrayFromImage(sitk.ReadImage(sv_sino_in_path))
    k = 0
    for i in range(len(sin_pre)):
        if i % scale == 0:
            sin_pre[i, :] = sin_original[k, :]
            k = k + 1
    # write dense-view sinogram and model
    sin_pre = sitk.GetImageFromArray(sin_pre)
    sitk.WriteImage(sin_pre, '{}/{}_sino_pre.nii'.format(dv_sino_out_path, num_dv))
