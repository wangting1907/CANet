# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:27:06 2024

@author: Administrator
"""

import os 
import yaml                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
import sys
import PIL
import torch
#import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as scio
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from math import exp
import math 
import random
import torch.nn as nn
from focal_frequency_loss import FocalFrequencyLoss as FFL
from scipy.ndimage import gaussian_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from skimage.transform import resize
import tomopy
import dxchange



def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_coordinate(*length):
    dim = len(length)
    if dim == 2:
        h, w = length
        # max_len = max(h, w)
        max_len = max(h, w)  # n
        grids = torch.linspace(-1, 1, steps=max_len)  # [n]
        # stack([n, n], [n, n]) -> [n, n, 2]
        coords = torch.stack(torch.meshgrid(grids, grids), dim=-1)

        if h >= w:
            minor = int((h - w) / 2)
            coords = coords[:, minor:w + minor]
        else:
            minor = int((w - h) / 2)
            coords = coords[minor:h + minor, :]

        coords = coords.numpy()
        coords = np.reshape(coords, (h*w, dim), order='F')
        coords = torch.tensor(coords)
     #   coords = coords.reshape(-1, dim)

    elif dim == 1:
        coords = torch.linspace(-1, 1, steps=length[0]).view(-1, 1)
    else:
        raise NotImplementedError(f"{dim}-D coordinates are not supported!")
    return coords


        
def save_yaml(dir, args, save_name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, save_name), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

def print_info(args):
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    #print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
def process_image(imgout):
    h,w,c=imgout.shape
    if np.count_nonzero(imgout < 0) > ((h*w*c) / 20):
        imgout -= np.median(imgout[imgout < 0])
    imgout[imgout <= 0] = 0
    return imgout


def flat_correct(projs, flats):
    results = []
    for i in range(0, projs.shape[0], 9):
        proj_slice = projs[i:i + 9]  
        if proj_slice.shape[0] == 9:  
            flat_slice = flats[i // 9]  
            #  -1 * log(proj / flat)
            result = -1 * np.log(proj_slice / flat_slice)
            results.append(result)
    sim = np.concatenate(results, axis=0)
    return sim


def background_remove(sim):
    c, h, w = sim.shape
    correct_data = np.zeros((c, 402, 331))
    correct_data1 = np.zeros((c, 331, 402))
    sim = np.transpose(sim, (0, 2, 1))
    
    # Background removal
    for i in range(c):
        ttpp = sim[i, :, :]
        if i <= 90:
            tmp1_cropped = ttpp[170:1050, 130:823]
        else:
            tmp1_cropped = ttpp[180:1060, 130:823]
        
        # Flip image horizontally
        flipped_img = np.fliplr(tmp1_cropped)  
        imgii = resize(flipped_img, (402, 331))
        correct_data[i, :, :] = imgii
    
    correct_data = np.transpose(correct_data, (0, 2, 1))
    
    for i in range(c):
        imgii = correct_data[i, :, :]
        
        # Smooth profiles using Gaussian filter
        top_profile = gaussian_filter(np.mean(imgii[:10, :], axis=0), sigma=2)
        bottom_profile = gaussian_filter(np.mean(imgii[-10:, :], axis=0), sigma=2.5)
        
        diff_profile = bottom_profile - top_profile
        
        # Create background
        height = imgii.shape[0]
        background = np.zeros_like(imgii)
        
        for jj in range(imgii.shape[1]):
            # Use a polynomial fit instead of linear interpolation
            x = np.linspace(0, height - 1, height)
            fit = np.polyfit(x, np.linspace(top_profile[jj], bottom_profile[jj], height), deg=2)
            background[:, jj] = np.polyval(fit, x)
        
        # Subtract background and set negative values to zero
        imgii_bkg_removed = imgii - background
        imgii_bkg_removed[imgii_bkg_removed < 0] = 0
        correct_data1[i, :, :] = imgii_bkg_removed
    
    return correct_data1   

def process_projection_data(data):
   
    thetas = data['angle'][:]
    tomo_data = data['img_tomo']  # 投影数据
    flat_data = data['img_bkg']   # 平场数据
    dark_data = data['img_dark']  # 暗场数据
    
    proj = tomopy.normalize(tomo_data, flat_data, dark_data)
    #print("After normalization: min =", np.min(proj), ", max =", np.max(proj))
    
    #clip
    proj = np.clip(proj, a_min=0.001, a_max=None)
    #print("After clipping: min =", np.min(proj), ", max =", np.max(proj))
    
    # -log
    proj = tomopy.minus_log(proj)
    return proj, thetas



def background_remove_real(sim):
    c, h, w = sim.shape
    correct_data = np.zeros((c, 426, 476))
    correct_data1 = np.zeros((c, 476, 426))
    sim = np.transpose(sim, (0, 2, 1))
    
    # Background removal
    for i in range(c):
        ttpp = sim[i, :, :]
        tmp1_cropped = ttpp[250:1000, 50:1000]
        # if i>=281:
        #     tmp1_cropped = np.flip(tmp1_cropped, axis=0)
        imgii = resize(tmp1_cropped, (426, 476))
        correct_data[i, :, :] = imgii
    
    correct_data = np.transpose(correct_data, (0, 2, 1))
    
    for i in range(c):
        imgii = correct_data[i, :, :]
        
        # Smooth profiles using Gaussian filter
        top_profile = gaussian_filter(np.mean(imgii[:10, :], axis=0), sigma=2.5)
        bottom_profile = gaussian_filter(np.mean(imgii[-10:, :], axis=0), sigma=2.5)
        diff_profile = bottom_profile - top_profile
        
        # Create background
        height = imgii.shape[0]
        background = np.zeros_like(imgii)
        
        for jj in range(imgii.shape[1]):
            # Use a polynomial fit instead of linear interpolation
            x = np.linspace(0, height - 1, height)
            fit = np.polyfit(x, np.linspace(top_profile[jj], bottom_profile[jj], height), deg=2)
            background[:, jj] = np.polyval(fit, x)
        
        # Subtract background and set negative values to zero
        imgii_bkg_removed = imgii - background
        imgii_bkg_removed[imgii_bkg_removed < 0] = 0
        correct_data1[i, :, :] = imgii_bkg_removed
    return correct_data1 
    
def load_and_process_data(fname):
    # Load the HDF5 file
    data_file = h5py.File(fname, 'r')
    # Process the projection data
    data, thetas = process_projection_data(data_file)
    # Process the image data
    data = process_image(data)
    # Remove background from the real data
    data = background_remove_real(data)
    return data, thetas

def process_projection_data_new(image_stack, n_flats=8, n_projs=724, bin_size=4):
    flats = image_stack[:n_flats]
    projs = image_stack[n_flats:n_flats+n_projs]
    darks = np.zeros_like(flats)

    # 投影分组平均
    if projs.shape[0] % bin_size != 0:
        raise ValueError(f"投影数量{projs.shape[0]} 无法被分组尺寸{bin_size}整除")
    
    binned_projs = projs.reshape(-1, bin_size, *projs.shape[1:]).mean(axis=1)

    # flat-corrected
    normalized = tomopy.normalize(binned_projs, flats, darks)

    # -log
    return tomopy.minus_log(normalized)   


def load_tiff_stack(folder_path):
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith(".tiff")]
    tiff_files.sort()
    image_stack = []
    for file_name in tiff_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"正在读取文件: {file_name}")
        

        img = Image.open(file_path)
        img_array = np.array(img)
        print(f"图像形状: {img_array.shape}")
        
        image_stack.append(img_array)
    return np.stack(image_stack, axis=0)


def background_remove_real_new(sim):
    c, h, w = sim.shape
    correct_data = np.zeros((c,150,230))  
    for i in range(c):
        imgii = sim[i, :, :]
        imgii = imgii[50:200, 70:300]
        # Smooth profiles using Gaussian filter
        top_profile = gaussian_filter(np.mean(imgii[:10, :], axis=0), sigma=2)
        bottom_profile = gaussian_filter(np.mean(imgii[-10:, :], axis=0), sigma=2)
        diff_profile = bottom_profile - top_profile  
        # Create background
        height = imgii.shape[0]
        background = np.zeros_like(imgii)
        
        for jj in range(imgii.shape[1]):
            # Use a polynomial fit instead of linear interpolation
            x = np.linspace(0, height - 1, height)
            fit = np.polyfit(x, np.linspace(top_profile[jj], bottom_profile[jj], height), deg=2)
            background[:, jj] = np.polyval(fit, x)
        
        # Subtract background and set negative values to zero
        imgii_bkg_removed = imgii - background
        imgii_bkg_removed[imgii_bkg_removed < 0] = 0
        correct_data[i, :, :] = imgii_bkg_removed
    
    return correct_data


class xcorr2(nn.Module):
    '''
    Correlates two images. Images must be of the same size.
    '''

    def __init__(self, zero_mean_normalize=True):
        super(xcorr2, self).__init__()
        self.InstanceNorm = nn.InstanceNorm2d(1, affine=False, track_running_stats=False)
        self.zero_mean_normalize = zero_mean_normalize


    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): Batch of Img1 of dimensions [B, C, H, W].
            x2 (torch.Tensor): Batch of Img2 of dimensions [B, C, H, W].
        Returns:
            scores (torch.Tensor): The correlation scores for the pairs. The output shape is [8, 1].
        """

        if x1.shape == x2.shape:
            scores = self.match_corr_same_size(x1, x2)
        else:
            scores = self.match_corr(x1, x2)
            
        return scores


    def match_corr_same_size(self, x1, x2):
        b, c, h, w = x2.shape

        if self.zero_mean_normalize:
            x1 = self.InstanceNorm(x1)
            x2 = self.InstanceNorm(x2)

        scores = torch.matmul(x1.view(b, 1, c*h*w), x2.view(b, c*h*w, 1))
        scores /= (h*w*c)
        
        return scores


    def match_corr(self, embed_ref, embed_srch):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].

        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.

        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        b, c, h, w = embed_srch.shape

        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.
        if self.zero_mean_normalize:
            embed_ref = self.InstanceNorm(embed_ref)
            embed_srch = self.InstanceNorm(embed_srch)

        # Has problems with mixed-precision training
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w), embed_ref, groups=b).float() 
        match_map /= (self.img_size*self.img_size*c)

        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)
        
        return match_map