import torch
import numpy as np
import time
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn.functional as F
import torch.cuda.amp
from torch.cuda.amp import autocast
from tomocupy_stream import GPURecRAM
from tomocupy_stream import find_center

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
    """
    Args:
        proj: numpy.ndarray of shape (c, h, w),Ungigned projection data 
        sim: numpy.ndarray of shape (c, h, w) Ground truth reference for simulation data. For real data, set sim=proj.
        dark: Dark field image for flat-field correction. If projections are already flat-corrected, provide an array of ones with shape (1, h, w). Otherwise, 
        provide the actual measured dark field.
        flat: Flat field image for flat-field correction. If projections are already flat-corrected, provide an array of ones with shape (1, h, w). Otherwise, 
        provide the actual measured flat field.
        theta: numpy.ndarray of shape (c,) Projection angles array, typically np.linspace(0, 180, c).astype('float32'), or actual acquisition angles.
        data: numpy.ndarray of shape (c, h, w), dtype float32 Pre-allocated container with same structure as proj for storing processing results.
        temporal_coord: angle index normalized [-1,1]
        temporal_model: siren
        ffl: loss function
        temperature: adjust last layer weight
    Returns:
        results
    """
    center_search_width = 10
    center_search_step = 0.5
    center_search_ind = data.shape[0]//2
    t_sim=sim.swapaxes(0,1)
    rotation_axis = find_center.find_center_vo(t_sim, dark, flat,
                                           ind=center_search_ind,
                                           smin=-center_search_width, 
                                           smax=center_search_width, 
                                           step=center_search_step)
    print('auto rotation axis',rotation_axis)
    cl = GPURecRAM.for_data_like(data=data,
                             dark=dark,
                             flat=flat,
                             ncz=8,  # chunk size for GPU processing (multiple of 2), 
                             rotation_axis=rotation_axis,  # rotation center
                             dtype="float32",  # computation type, note  for float16 n should be a power of 2
                             reconstruction_algorithm='fourierrec',
                             fbp_filter='parzen', 
                             minus_log=False)
    err = np.zeros((proj.shape[0]))
    err_list = []
    list_loss_iteration = []
    psnr_list = []
    recon_list = []
    mse_list = []
    sx = np.zeros((proj.shape[0]))
    sy = np.zeros((proj.shape[0]))
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
        ref_torch = ((re_sim / np.max(re_sim))*(np.max(re_sim) - np.min(re_sim)))
        ref_torch = torch.tensor(ref_torch, device=device, dtype=torch.float32)
        re_mse = torch.sum((pred_torch - sim_torch) ** 2) / torch.sum(sim_torch ** 2)
        mse_list.append(re_mse.item())
        loss_list = []
        temporal_model.train()
        for epoch in range(1, epochs + 1):
            with autocast():  
                output = temporal_model(temporal_coord)
                output = output/temperature
                motion_output = batch_affine_transform(pred_torch, output,mode)
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
    end_time=time.time()   
    print(f"Elapsed time: {(end_time - start_time):.2f} seconds")

    results = {
        'err_list': err_list,
        'psnr_list': psnr_list,
        'mse_list': mse_list,
        'list_loss_iteration': list_loss_iteration,
        'pred': pred_torch.cpu().numpy(),
        'final_shift': shift
    }
    return results
# # 使用示例
# if __name__ == "__main__":
#     ## defined variable
#     # pred, sim, temporal_coord, dark, flat, theta, temporal_model, ffl
    
#     results = run_temporal_reconstruction(
#         pred=y_noisy,
#         sim=sim,
#         dark=dark,
#         flat=flat,
#         theta=theta,
#         temporal_coord=temporal_coord,
#         temporal_model=temporal_model,
#         ffl=ffl,
#         epochs=200,
#         lr=1e-4,
#         wd=0.0,
#         iters=10
#     )
    
#     # result
#     print(f"Final PSNR: {results['psnr_list'][-1]:.2f}")
#     print(f"Total time: {sum(results['time_list']):.2f} seconds")