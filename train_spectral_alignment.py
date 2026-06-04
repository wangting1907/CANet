import yaml
import argparse
import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.io as scio

from inr import models
from focal_frequency_loss import FocalFrequencyLoss as FFL
from utils import *
from proprecessing import *

def apply_batch_affine_transform(images, shifts, scaling_factors, mode='bilinear', scaling=True):
    device = images.device
    B, H, W = images.shape
    theta = torch.zeros(B, 2, 3, device=device)
    if scaling:
        # scaling_factors 形状期望为 (B,) 或 (B,2)
        s = scaling_factors
        theta[:, 0, 0] = s if s.dim() == 1 else s[:, 0]
        theta[:, 1, 1] = s if s.dim() == 1 else s[:, 1]
    else:
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0

    theta[:, :, 2] = shifts
    images = images.unsqueeze(1)
    grid = F.affine_grid(theta, images.size(), align_corners=True)
    transformed = F.grid_sample(images, grid, mode=mode, padding_mode='zeros', align_corners=True)
    return transformed.squeeze(1)

def main(cli_args):
    with open(cli_args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg['training']['seed'])
    log_dir = os.path.join(cfg['paths']['log_dir_root'], f"energy_{cli_args.energy_index}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    txt_file = Path(cfg['paths']['txt_dir'])
    energies, refs, collects = parse_scan_file(txt_file)
    energy_idx = cli_args.energy_index
    print(f"--- Loading Energy: {energies[energy_idx]} eV ---")
    
    flats, projs, thetas = load_energy_index(energy_idx, refs, collects)
    sim = flat_correct(projs, flats)
    sim = process_image(sim)
    sim = background_remove(sim) 

    target_data = scio.loadmat(cfg['paths']['highest_proj'])
    target = target_data.get("pred")
    assert sim.shape == target.shape, f"Shape mismatch: Sim {sim.shape} vs Target {target.shape}"

    n_bands = target.shape[0]
    temporal_coord = get_coordinate(n_bands)
    k = 3 if cfg['model']['scaling'] else 2
    lam = cfg['loss']['lambda_corr'] if cfg['model']['scaling'] else 0

    temporal_params = {
        'nonlin': cfg['model']['type'],
        'in_features': 1,
        'out_features': k,
        'hidden_features': cfg['model']['hidden_features'],
        'hidden_layers': cfg['model']['hidden_layers'],
        'outermost_linear': True
    }
    temporal_model = models.get_INR(**temporal_params)
    device = torch.device('cuda' if cfg['training']['cuda'] and torch.cuda.is_available() else 'cpu')
    X_torch = torch.tensor(sim, dtype=torch.float32).to(device)
    ref_torch = torch.tensor(target, dtype=torch.float32).to(device)
    temporal_coord = temporal_coord.to(device)
    temporal_model = temporal_model.to(device)
    correlate = xcorr2(zero_mean_normalize=True).to(device)
    ffl = FFL(loss_weight=cfg['loss']['gamma1'], alpha=cfg['loss']['gamma2'])
    optimizer = torch.optim.Adam(temporal_model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['wd'])
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.9)
    print("Starting optimization...")
    start_time = time.time()
    for epoch in range(1, cfg['training']['epochs'] + 1):
        output = temporal_model(temporal_coord)
        
        if cfg['model']['scaling']:
            aff = (output[:, 0:1] + 1) * 2  
            trans = torch.tanh(output[:, 1:3])
        else:
            aff = torch.ones_like(output[:, 0:1])
            trans = torch.tanh(output[:, 0:2])

        motion_output = apply_batch_affine_transform(X_torch, trans, aff.squeeze(), scaling=cfg['model']['scaling'])
        loss_ffl = ffl(motion_output.unsqueeze(1), ref_torch.unsqueeze(1))
        loss_corr = lam * torch.mean(1 - correlate(motion_output.unsqueeze(1), ref_torch.unsqueeze(1)))
        total_loss = loss_ffl + loss_corr

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{cfg['training']['epochs']} - Loss: {total_loss.item():.6f}")
    pred_np = motion_output.detach().cpu().numpy()
    result_base = cfg['paths']['result_dir']
    os.makedirs(result_base, exist_ok=True)
    
    save_name = os.path.join(result_base, f"alignment_{int(energies[energy_idx])}.npy")
    np.save(save_name, pred_np)
    
    print(f"Saved: {save_name} | Elapsed Time: {time.time() - start_time:.2f}s")
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectral Alignment')
    parser.add_argument('--config', type=str, default='config_spectral.yaml')
    parser.add_argument('--energy_index', type=int, required=True, help='Current energy scan index')
    args = parser.parse_args()

    main(args)