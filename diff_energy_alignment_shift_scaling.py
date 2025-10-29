import argparse
import os
import time
from datetime import datetime
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.tensorboard import SummaryWriter
from inr import models
from utils import *
import torch
import skimage
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from inr import models
from math import exp
import math 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from focal_frequency_loss import FocalFrequencyLoss as FFL
import scipy.io as scio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import astra
from tomocupy_stream import GPURecRAM
from tomocupy_stream import find_center
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from proprecessing import *

def apply_batch_affine_transform(images, shifts, scaling_factors, mode='bilinear', scaling=True):
    device = images.device
    B, H, W = images.shape
    # 准备仿射矩阵
    theta = torch.zeros(B, 2, 3, device=device)  
    # 处理缩放
    if scaling:
        if scaling_factors.dim() == 1:
            # 各向同性缩放 (B,)
            theta[:, 0, 0] = scaling_factors
            theta[:, 1, 1] = scaling_factors
        else:
            # 各向异性缩放 (B,2)
            theta[:, 0, 0] = scaling_factors[:, 0]
            theta[:, 1, 1] = scaling_factors[:, 1]
    else:
        # 无缩放，使用单位矩阵
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
    # 添加平移
    theta[:, :, 2] = shifts
    # 应用变换
    images = images.unsqueeze(1)  # (B, 1, H, W)
    grid = F.affine_grid(theta, images.size())
    transformed = F.grid_sample(images, grid, mode=mode, padding_mode='zeros',align_corners=True)
    return transformed.squeeze(1)  # (B, H, W)
def main(args):
    args.datetime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    # fix seed
    set_seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # configure(args.log_dir)

    writer = SummaryWriter(log_dir=args.log_dir)
    save_yaml(args.log_dir, vars(args), 'config.yml')
    print_info(args)

    # load projs and flats data
    txt_file = Path(args.txt_dir)
    energies, refs, collects = parse_scan_file(txt_file)
    print(energies)
    energy_index=args.energy_index
    print("Loading energy %feV"%(energies[energy_index]))
    flats, projs, thetas = load_energy_index(energy_index, refs, collects)
    ali_four_digits = energies[energy_index]
    print(f'alignment energy "{ali_four_digits}"')
    #background_correction   
    sim=flat_correct(projs,flats)
    assert sim.shape == projs.shape
    sim= process_image(sim)            ##remove negative
    sim=background_remove(sim)         ##remove background and resize
    
    # load highest or middest projection data
    imgs=scio.loadmat(args.highest_proj)
    target=imgs.get("pred")
    last_four_digits = args.highest_proj[-8:-4]
    print(f'target energy "{last_four_digits}"')
    assert sim.shape == target.shape
    n_bands = target.shape[0]
    temporal_coord = get_coordinate(n_bands)  # 1D # [C]
    if args.scaling:
        X_torch=torch.tensor(sim).to(torch.float32)
        ref_torch=torch.tensor(target).to(torch.float32)
        k=3
        args.epochs=200
        lam=0.1
    else:
        X_torch=torch.tensor(sim).to(torch.float32)
        ref_torch=torch.tensor(target).to(torch.float32)
        k=2
        args.epochs=200
        lam=0
    if args.cuda:
        X_torch=X_torch.cuda() 
        ref_torch=ref_torch.cuda()
        temporal_coord=temporal_coord.cuda()
        correlate=xcorr2(zero_mean_normalize=True).cuda()
    pred=torch.zeros_like(X_torch)
    ffl = FFL(loss_weight=args.gamma1, alpha=args.gamma2)
    err = np.zeros((pred.shape[0]))
    err_list=[]
    loss_list=[]
    ## Shift arrays
    sx = np.zeros((pred.shape[0]))
    sy = np.zeros((pred.shape[0]))
    temporal_params = {'nonlin': args.inr, 'in_features': 1, 'out_features': k,'hidden_features': args.temporal_hidden_features,
                       'hidden_layers': args.temporal_hidden_layers, 'outermost_linear': True}
    temporal_model = models.get_INR(**temporal_params)
    optimizer = torch.optim.Adam([{'params': temporal_model.parameters()}], lr=args.lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.9) 
    temporal_model.train()
    temporal_model = temporal_model.cuda()
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        output = temporal_model(temporal_coord)
        aff=output[:,0:1]
        trans=output[:,1:3]
        aff = (aff + 1)*2   # 应用 scaling
        # aff = torch.diag(aff.squeeze())
        trans=torch.tanh(trans)
        motion_output = apply_batch_affine_transform(X_torch, trans, aff.squeeze(), mode='bilinear', scaling=args.scaling)
        loss = ffl(motion_output.unsqueeze(1), ref_torch.unsqueeze(1))+lam*torch.mean((1-correlate (motion_output.unsqueeze(1), ref_torch.unsqueeze(1))))
        optimizer.zero_grad(set_to_none=True)  # 更快的清零
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        loss_list.append(loss.item()) 
    pred=motion_output.squeeze().detach().cpu().numpy() 
    shift=output.squeeze().detach().cpu().numpy() 
    end_time = time.time()
    elapsed_time = end_time - start_time
    # 在循环结束后，保存 affine 和 trans 参数
    affine_params = aff.detach().cpu().numpy()  # 转为 NumPy 数组
    trans_params = trans.detach().cpu().numpy()   # 转为 NumPy 数组
    writer.add_scalar('all_time',elapsed_time )
    writer.close()
    result_dir = os.path.join(args.result_dir, f"alignment_{int(ali_four_digits)}")
    np.save(result_dir,pred) 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='alignment')
    parser.add_argument('--txt_dir', type=str, help='txt file path')
    parser.add_argument('--energy_index', type=int, help='energy_index')
    parser.add_argument('--highest_proj', type=str, help='data file path')
    parser.add_argument('--snr', type=int, help='SNR parameter for adding noise')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')

    parser.add_argument('--wd', type=float, default=0., metavar='WD',
                        help='weight decay (default: 0.0)')

    parser.add_argument('--epochs', type=int, default=1000, metavar='epochs',
                        help='training iterations (default: 1000)')

    parser.add_argument('--eval_freq', type=int, default=200)

    parser.add_argument('--scaling', action='store_true', default=False, help='Enable scaling')
    
    parser.add_argument('--cuda', action='store_true',
                        help='use cuda')

    parser.add_argument('--log_dir', default='./runs', type=str, metavar='PATH',
                        help='where checkpoints and logs to be saved (default: ./runs)')
  
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (defualt: 42)')

    parser.add_argument('--inr', type=str, default='siren', help='inr model')
    parser.add_argument('--temporal_hidden_features', type=int, default=256, help='hidden_features (default: 12)')
    parser.add_argument('--temporal_hidden_layers', type=int, default=3, help='hidden layers for temporal model')
    parser.add_argument('--gamma1', type=float, default=3,help='gamma1 hyper-parameter')
    parser.add_argument('--gamma2', type=float, default=1,help='gamma2 hyper-parameter')
    parser.add_argument('--result_dir', type=str, default='./alignment_result/',help='data file path')
    args = parser.parse_args()

    main(args)
