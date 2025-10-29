import pdb
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


from .utils import build_montage, normalize
class ReLULayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with ReLU non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return nn.functional.relu(self.linear(input))
    
class PosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            #assert fn_samples is not None
            fn_samples = sidelength
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = 4

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

class Msi2Delta(nn.Module):
    def __init__(self, input_c, output_c, ngf=64, n_res=3, useSoftmax=True):
        super(Msi2Delta, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf*2, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*2, ngf*4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*4, ngf*8, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*8, output_c, 1, 1, 0),
            nn.ReLU()
            # nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=1)
        self.usefostmax = useSoftmax

    def forward(self, x):
        if self.usefostmax == True:
            return self.softmax(self.net(x))
        elif self.usefostmax == False:
            return self.net(x)
    
class INR(nn.Module):
    def __init__(self, in_features,
                 hidden_features, hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=True, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        if pos_encode:
            self.positional_encoding = PosEncoding(in_features=2,
                                                   sidelength=sidelength,
                                                   fn_samples=fn_samples,
                                                   use_nyquist=use_nyquist)
            in_features = self.positional_encoding.out_dim
        ngf=64  
        useSoftmax=True
        self.net1 = nn.Sequential(
              nn.Conv2d(4, ngf*2, 1, 1, 0),
              nn.LeakyReLU(0.2, True),
              nn.Conv2d(ngf*2, ngf*4, 1, 1, 0),
              nn.LeakyReLU(0.2, True),
              nn.Conv2d(ngf*4, ngf*8, 1, 1, 0),
              nn.LeakyReLU(0.2, True),
              nn.Conv2d(ngf*8, 8, 1, 1, 0),
              nn.ReLU()
          )
        self.softmax = nn.Softmax(dim=1)
        self.usefostmax = useSoftmax
        
        
        self.pos_encode = pos_encode
        
        self.complex = False
        self.nonlin = ReLULayer
            
        self.net = []
        self.net.append(self.nonlin(in_features+8, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            if self.complex:
                dtype = torch.cfloat
            else:
                dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
                        
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        self.net = nn.Sequential(*self.net)
        
    
    def forward(self, coords,x_hr):
        if self.usefostmax == True:
            x= self.softmax(self.net1(x_hr))
        elif self.usefostmax == False:
            x= self.net(x_hr)
        out=x.squeeze()
        out=out.permute(2,1,0)
        out= out.reshape(336*336,8)
        out=out.detach()
        out=(out-0.5)*2
        if self.pos_encode:
            coords = self.positional_encoding(coords)
        coords=coords.squeeze()
        coord1= torch.cat((coords, out), dim=1)
        output = self.net(coord1)                 
        return output
 
 