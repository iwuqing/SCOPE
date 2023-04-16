# ----------------------------------------------#
# Pro    : SCOPE
# File   : eval.py
# Date   : 2023/4/17
# Author : Qing Wu                        
# Email  : wuqing@shanghaitech.edu.cn        
# ----------------------------------------------#
import utils
import SimpleITK as sitk
import numpy as np

if __name__ == '__main__':
    gt = sitk.GetArrayFromImage(sitk.ReadImage('data/gt_img_0_80.nii'))
    recon = sitk.GetArrayFromImage(sitk.ReadImage('output/img/scope_sin_pre_80_90.nii'))

    print('PSNR:', utils.psnr(image=recon, ground_truth=gt), 'SSIM:', utils.ssim(image=recon, ground_truth=gt))