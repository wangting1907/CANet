import os
import argparse
import yaml
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.io as scio
from skimage.metrics import peak_signal_noise_ratio
from tomocupy_stream import GPURecRAM, find_center
from inr import models 
from focal_frequency_loss import FocalFrequencyLoss as FFL
from utils import get_coordinate, set_seed

def batch_affine_transform(images, shifts, mode='bilinear'):
    """
    Args:
        images: (B, H, W)  input
        shifts: (B, 2)     affine parameter
    Returns:
        transformed: (B, H, W) aligned image
    """
    device = images.device
    B, H, W = images.shape
    images = images.unsqueeze(1)  
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = 1  
    theta[:, 1, 1] = 1
    theta[:, :, 2] = shifts  
    grid = F.affine_grid(theta, (B, 1, H, W))
    transformed = F.grid_sample(images, grid, mode=mode, padding_mode='zeros')
    return transformed.squeeze(1)  # (B, H, W)

def projection_reprojection_alignment(proj, sim, dark, flat, theta, data, temporal_coord, temporal_model, ffl, temperature, epochs=200, lr=1e-4, wd=0.0, 
    iters=10, k=2, eval_freq=100, cuda=True, mode='bicubic'):
    center_search_width = 10
    center_search_step = 0.5
    center_search_ind = data.shape[0]//2
    t_sim = sim.swapaxes(0, 1)
    
    rotation_axis = find_center.find_center_vo(t_sim, dark, flat,
                                               ind=center_search_ind,
                                               smin=-center_search_width, 
                                               smax=center_search_width, 
                                               step=center_search_step)
    print('auto rotation axis', rotation_axis)
    
    cl = GPURecRAM.for_data_like(data=data,
                                 dark=dark,
                                 flat=flat,
                                 ncz=8,  
                                 rotation_axis=rotation_axis,  
                                 dtype="float32",  
                                 reconstruction_algorithm='fourierrec',
                                 fbp_filter='parzen', 
                                 minus_log=False)
                                 
    err_list = []
    list_loss_iteration = []
    psnr_list = []
    mse_list = []
    
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
    max_sim = np.max(sim)
    sim_torch = torch.tensor(sim, device=device, dtype=torch.float32)
    temporal_coord = temporal_coord.to(device)
    pred_torch = torch.tensor(proj, device=device, dtype=torch.float32)
    temporal_model = temporal_model.to(device)
    
    optimizer = torch.optim.Adam(
        [{'params': temporal_model.parameters()}], 
        lr=lr, 
        weight_decay=wd
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.9)
    
    start_time = time.time()
    for n in range(iters):
        obj = cl.recon_all(pred_torch.cpu().numpy().swapaxes(0, 1), dark, flat, theta)
        re_sim = cl.proj_all(obj, theta).swapaxes(0, 1)
        ref_torch = ((re_sim / np.max(re_sim)) * (np.max(re_sim) - np.min(re_sim)))
        ref_torch = torch.tensor(ref_torch, device=device, dtype=torch.float32)
        re_mse = torch.sum((pred_torch - sim_torch) ** 2) / torch.sum(sim_torch ** 2)
        mse_list.append(re_mse.item())
        loss_list = []
        temporal_model.train()
        for epoch in range(1, epochs + 1):
            with autocast():  
                output = temporal_model(temporal_coord)
                output = output / temperature
                motion_output = batch_affine_transform(pred_torch, output, mode)
                loss = ffl(motion_output.unsqueeze(1), ref_torch.unsqueeze(1))     
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            loss_list.append(loss.item())
            
        list_loss_iteration.append(loss_list) 
        
        with torch.no_grad():
            pred_torch = motion_output.detach()
            shift = output.squeeze().detach().cpu().numpy()
            tmp_psnr = peak_signal_noise_ratio(
                sim / max_sim, 
                pred_torch.cpu().numpy() / pred_torch.max().item()
            )
            tmp_err = np.linalg.norm(shift)   
            print(f'[{n}/{iters}], error: {tmp_err:.4f}, all_psnr: {tmp_psnr:.2f}') 
            err_list.append(tmp_err)
            psnr_list.append(tmp_psnr)
            save_dir = f'mat_results_{mode}' 
            os.makedirs(save_dir, exist_ok=True)
            save_data = {
                f'obj_{mode}': obj,                             
                f'pred_torch_{mode}': pred_torch.cpu().numpy()  
            }
            mat_filename = os.path.join(save_dir, f'tomo_0089_iter_{n:03d}.mat')
            scio.savemat(mat_filename, save_data)
            print(f"Saved mat file to {mat_filename}")

    end_time = time.time()   
    print(f"Elapsed time: {(end_time - start_time):.2f} seconds")
    
    return {
        'err_list': err_list,
        'psnr_list': psnr_list,
        'mse_list': mse_list,
        'list_loss_iteration': list_loss_iteration,
        'pred': pred_torch.cpu().numpy(),
        'final_shift': shift
    }

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='CANet Tomo Alignment (Strict)')
    parser.add_argument('--config', type=str, default='config_tomo.yaml')
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['training']['seed'])
    data_path = args.data_path if args.data_path else cfg['paths']['data_path']
    sim_path = cfg['paths']['sim_path']
    
    proj_data = np.load(data_path).astype('float32')
    sim_data = np.load(sim_path).astype('float32')
    c, h, w = proj_data.shape

    theta = np.deg2rad(np.linspace(cfg['geometry']['theta_start'], cfg['geometry']['theta_end'], c)).astype('float32')
    
    dark = np.zeros([1, h, w], dtype='float32')
    flat = np.ones([1, h, w], dtype='float32')
    data_container = np.zeros([c, h, w], dtype='float32').swapaxes(0, 1)
    
    temporal_coord = get_coordinate(c)
    
    temporal_model = models.get_INR(
        nonlin=cfg['model']['type'], 
        in_features=1,
        out_features=cfg['model']['k_out'],
        hidden_features=cfg['model']['hidden_features'],
        hidden_layers=cfg['model']['hidden_layers'], 
        outermost_linear=True
    )

    ffl = FFL(loss_weight=cfg['loss']['ffl_weight'], alpha=cfg['loss']['alpha'])
    results = projection_reprojection_alignment(
        proj=proj_data,
        sim=sim_data,
        dark=dark,
        flat=flat,
        theta=theta,
        data=data_container,
        temporal_coord=temporal_coord,
        temporal_model=temporal_model,
        ffl=ffl,
        temperature=cfg['model']['temperature'],
        epochs=cfg['training']['epochs'],
        lr=cfg['training']['lr'],
        wd=cfg['training']['wd'],
        iters=cfg['training']['iters'],
        k=cfg['model']['k_out'],
        mode=cfg['training']['mode'],
        cuda=cfg['training']['cuda']
    )
    save_base = cfg['paths']['result_dir'] if 'result_dir' in cfg['paths'] else "./"
    os.makedirs(save_base, exist_ok=True)
    scio.savemat(os.path.join(save_base, "tomo_aligned_final.mat"), results)

if __name__ == '__main__':
    main()