# SCOPE

This repository contains the PyTorch implementation of our manuscript "Self-Supervised Coordinate Projection Network for Sparse-View Computed Tomography". [[ArXiv](https://arxiv.org/abs/2209.05483)] [[IEEE Xplore](https://ieeexplore.ieee.org/document/10143286)]

## 1.  Main Running Environment

To run this project, you will need the following packages:
- PyTorch
- tinycudann
- SimpleITK, tqdm, numpy, and other packages.

## 2. File Tree

```text
SCOPE
│  config.json          # configuration file.
│  dataset.pyc
│  eval.py              # evaluates the reconstruted CT result.
│  README.md
│  reprojection.py      # generates DV sinogram via the fine-trained MLP.
│  scope.pyc
│  train.py             # trains the MLP network.
│  utils.py
│
├─data
│      90_img.nii       # CT image by FBP on the SV sinogram (90_sino.nii).
│      90_sino.nii      # SV sinogram (input data).
│      gt_img.nii       # GT CT image by FBP on the GT DV sinogram (gt_sino.nii).
│      gt_sino.nii      # GT DV sinogram (reference data).
│
├─model
│      checkpoint.pth   # pre-trained model for SV sinogram (90_sino.nii).
│
├─output
│  ├─img
│  │      scope_recon.nii     # Our reconstructed result.
│  │
│  └─sino
│          720_sino_pre.nii   # DV sinogram generated by SCOPE.
│
└─script_matlab
        gene_angle.m
        gene_img.m      # matlab script for FBP algorithm.
```

## 3. Training and Re-projection

To train the model from scratch, navigate to `./` and run the following command in your terminal:
```shell
python train.py
```
This will train the model for the input sinogram (`90_sino.nii`). The pre-trained model will be stored in `./model`.

Next, go to `./` and run the following command in your terminal for reprojting DV sinogram:
```shell
python reprojection.py
```
This will generate the DV sinogram, which will be stored in `output/sino`.

Finally, navigate to `./script_matlab` and use MATLAB to run gene_img.m to recover the final CT image, which will be stored in `./output/img`.

## 4. Evaluation

To qualitatively evalute the result, navigate to `./` and run the following comman in your terminal:
```shell
python eval.py
```
This will compute PSNR and SSIM values for the reconstruced image (`./output/img/scope_recon.nii`). PSNR and SSIM are respectively 40.45 dB and 0.9794 for our provied result. 

## 5. License

This code is available for non-commercial research and education purposes only. It is not allowed to be reproduced, exchanged, sold, or used for profit.

## 6. Citation

If you find our work useful in your research, please cite:

```
@ARTICLE{10143286,
  author={Wu, Qing and Feng, Ruimin and Wei, Hongjiang and Yu, Jingyi and Zhang, Yuyao},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Self-Supervised Coordinate Projection Network for Sparse-View Computed Tomography}, 
  year={2023},
  volume={9},
  number={},
  pages={517-529},
  doi={10.1109/TCI.2023.3281196}}
```
