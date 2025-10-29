Data-driven deformation correction in X-ray spectro-tomography with implicit neural networks
====

Installation
-------
1. Clone the repository:
   
```
conda create -n CANet_alignment python=3.12.7
conda activate CANet_alignment
git clone https://github.com/wangting1907/CANet
cd CANet
```
2. Install packages:

```
conda install -c httomo tomophantom     
conda install -c conda-forge tomopy
pip install -r requirements.txt
```

3.  Install the tomocupy-stream:

```
cd tomocupy-stream    
pip install cupy scikit-build swig cmake h5py pywavelets matplotlib notebook
pip install .
```



Usage
-------
### 1. Input Data Preparation
   
   The data utilized in this work are from [tomobank](https://tomobank.readthedocs.io/en/latest/), specifically the sample with [tomo_00089](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.XANES.html).

   #### Step 1. Tomography data: input should be in an .h5 file or .npy file format containing the following datastes:

  | Dataset Name | Description | Shape |
  | :--- | :--- | :--- |
  | **`projections`** | Projection data (sinograms) | `(num_angles, height, width)` |
  | **`flat`** | Flat-field reference images | `(num_flat, height, width)` or `(1, height, width)` |
  | **`dark`** | Dark-field reference images | `(num_dark, height, width)` or `(1, height, width)` |

    • For tomo_00089 (8346eV), refer to CANet/tomo_alignment_real_data_0089.ipynb
    
    • For simulated data,  refer to CANet/tomo_alignment_simulation.ipynb
   
   #### Step 2. Spectral data: Download data [tomo_00089](https://tomobank.readthedocs.io/en/latest/source/) into CANet/Data/XANES_00089
   

### 2.   Running (Alignment)
   CANet include two steps:
   #### Step 1: Tomography alignment (Jupyter Notebook)

    • tomo_alignment_simulation.ipynb -- For simulated experiments 

    • tomo_alignment_real_data_0089.ipynb -- For real experiments

   #### Step 2: Spectral alignment (Mult-energies)

    • Save aligned projections from Step 1 to: ./Data/XANES_00089/highest_energy/
    
    • Run the script from the terminal 
```
sh run_shift.sh
```

### 3.   Reconstruction 
    
     • real_data_0089_recon.ipynb --Reconstruct the 3D volume at different energy

Citation
-------
If you find our work useful in your research, please cite:




