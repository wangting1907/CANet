#!/usr/bin/env python
# -*- coding: utf-8 -*-

# *************************************************************************** #
#                  Copyright © 2022, UChicago Argonne, LLC                    #
#                           All Rights Reserved                               #
#                         Software Name: Tomocupy_stream                             #
#                     By: Argonne National Laboratory                         #
#                                                                             #
#                           OPEN SOURCE LICENSE                               #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
#    this list of conditions and the following disclaimer.                    #
# 2. Redistributions in binary form must reproduce the above copyright        #
#    notice, this list of conditions and the following disclaimer in the      #
#    documentation and/or other materials provided with the distribution.     #
# 3. Neither the name of the copyright holder nor the names of its            #
#    contributors may be used to endorse or promote products derived          #
#    from this software without specific prior written permission.            #
#                                                                             #
#                                                                             #
# *************************************************************************** #
#                               DISCLAIMER                                    #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS         #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT           #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS           #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT    #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,      #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED    #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR      #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF      #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING        #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS          #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                #
# *************************************************************************** #


from tomocupy_stream import cfunc_linerec
from tomocupy_stream import cfunc_linerecfp16
import cupy as cp


class LineRec():
    """Backprojection by summation over lines"""

    def __init__(self, nproj, ncproj, nz, ncz, n, dtype):
        self.nproj = nproj
        self.ncproj = ncproj
        self.nz = nz
        self.ncz = ncz
        self.n = n
        self.dtype = dtype

        if dtype == 'float16':
            self.fslv = cfunc_linerecfp16.cfunc_linerec(
                nproj, nz, n, ncproj, ncz)
        else:
            self.fslv = cfunc_linerec.cfunc_linerec(nproj, nz, n, ncproj, ncz)

    def backprojection(self, f, data, theta,  stream=0, lamino_angle=0, sz=0):
        phi = cp.float(cp.pi/2+(lamino_angle)/180*cp.pi)
        self.fslv.backprojection(
            f.data.ptr, data.data.ptr, theta.data.ptr, phi, sz, stream.ptr)

    def backprojection_try(self, f, data, theta, sh, stream=0, lamino_angle=0, sz=0):
        phi = cp.float(cp.pi/2+(lamino_angle)/180*cp.pi)
        self.fslv.backprojection_try(
            f.data.ptr, data.data.ptr, theta.data.ptr, sh.data.ptr, phi, sz, stream.ptr)

    def backprojection_try_lamino(self, f, data, theta, sh, stream=0, lamino_angle=0, sz=0):
        phi = (cp.pi/2+(lamino_angle+sh)/180*cp.pi).astype('float32')
        self.fslv.backprojection_try_lamino(
            f.data.ptr, data.data.ptr, theta.data.ptr, phi.data.ptr, sz, stream.ptr)
