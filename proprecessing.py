# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:38:28 2024

@author: Administrator
"""

from pathlib import Path
import dxchange
import numpy as np
import tomopy

# Loads tomo_00089 dataset from Tomobank for processing
# data was taken in following patter...
# set energy
# take 10 reference flats
# take [-90, -1] projects in 1 deg interval
# take 10 reference flats
# take [0, 89] projects in 1 deg interval
# repeat for each energy

def load_xrm_list(xrm_list):
    data_stack = None
    metadatas = []
    for i, filename in enumerate(xrm_list):
        data, metadata = dxchange.read_xrm(str(filename))
        if data_stack is None:
            data_stack = np.zeros((len(xrm_list),)+data.shape, data.dtype)
        data_stack[i] = data
        metadatas.append(metadata)
    return data_stack, metadatas


def parse_scan_file(txt_file):
    energies = []
    refs = []
    collects = []
    with open(txt_file, "r") as f:
        for line in f.readlines():
            if line.startswith("sete "):
                energies.append(float(line[5:]))
                refs.append([])
                collects.append([])
            elif line.startswith("collect "):
                filename = line[8:].strip()
                if "_ref_" in filename:
                    refs[-1].append(Path(txt_file).parent / filename)
                else:
                    collects[-1].append(Path(txt_file).parent / filename)
    return energies, refs, collects


def load_energy_index(energy_index, refs, collects):
    flats, _ = load_xrm_list(refs[energy_index])
    projs, metadatas = load_xrm_list(collects[energy_index])
    thetas = [metadata['thetas'][0] for metadata in metadatas]
    return flats, projs, thetas


if __name__ == "__main__":
    txt_file = Path("tomo_00089/AC3_C4p6_3DXANES/AC3_C4p6_3DXANES_TOMO-XANES.txt")
    energies, refs, collects = parse_scan_file(txt_file)
    print(energies)
    energy_index = 0
    print("Loading energy %feV"%(energies[energy_index]))
    flats, projs, thetas = load_energy_index(energy_index, refs, collects)
    print(projs.shape)
    print(np.min(projs))
    print(np.max(projs))
    #np.save('./data/tomo_0089_7728.npy', projs)
    #np.save('theta.npy',thetas)
    # the flats can be split into two groups for better flat field correction
    flats1 = flats[:10]
    flats2 = flats[10:]
    # same for projections...
    projs1 = projs[:90]
    projs2 = projs[90:]
