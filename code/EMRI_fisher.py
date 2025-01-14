import numpy as np
import cupy as cp
import pandas as pd
import time
import h5py
from scipy.signal.windows import tukey
from lisatools.sensitivity import noisepsd_AE,noisepsd_T
from lisatools.diagnostic import *
import sys
from tqdm import tqdm as tqdm
from few.waveform import GenerateEMRIWaveform, Pn5AAKWaveform
from fastlisaresponse import ResponseWrapper
from stableemrifisher.fisher import StableEMRIFisher
import argparse 
import seaborn as sns
from pathlib import Path
import logging
import concurrent.futures
import time
from timeout_decorator import timeout, TimeoutError
import matplotlib.pyplot as plt
from collections import defaultdict

# Clear all CuPy memory pools
cp._default_memory_pool.free_all_blocks()
cp._default_pinned_memory_pool.free_all_blocks()

# Synchronize GPU to ensure all tasks are completed
cp.cuda.runtime.deviceSynchronize()

# Configure logging to capture skipped sources
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def check_memory():
    free_memory, total_memory = cp.cuda.Device(0).mem_info

    print(f"Total GPU Memory: {total_memory / 1e9:.2f} GB")

    print(f"Free GPU Memory: {free_memory / 1e9:.2f} GB")
    print(f"Used memory = {(total_memory - free_memory)/1e9:.2f} GB")

@timeout(60)
def compute_fisher_matrix(j, param, args, outdir, param_names):
    try:
        # Initialization
        fish = StableEMRIFisher(
            np.float64(param.M[j] * (1 + param.z[j])),
            np.float64(param.mu[j] * (1 + param.z[j])),
            np.float64(param.a[j]),
            np.float64(param.p0[j]),
            np.float64(param.e0[j]),
            np.float64(param.Y0[j]),
            np.float64(param.dist[j]),
            np.float64(param.qS[j]),
            np.float64(param.phiS[j]),
            np.float64(param.qK[j]),
            np.float64(param.phiK[j]),
            np.float64(param.Phi_phi0[j]),
            np.float64(param.Phi_theta0[j]),
            np.float64(param.Phi_r0[j]),
            dt=args.delta_t,
            T=args.T_lisa,
            EMRI_waveform_gen=EMRI_TDI,
            param_names=param_names,
            stats_for_nerds=True,
            stability_plot=False,
            Ndelta=16,
            use_gpu=True,
            filename=outdir,
            live_dangerously=False, 
            der_order=2
        )

        # Execution
        fisher_matrix = fish()
        covariance_matrix = np.linalg.inv(fisher_matrix)

        return fisher_matrix, covariance_matrix 

    except Exception as e:
        print("ValueError", e)
        return None, None

parser = argparse.ArgumentParser()

parser.add_argument("nmodel", type = int, help = "model number")
parser.add_argument("delta_t", type = int, help = "sample rate in s", default = 5)
parser.add_argument("t_gw", type = int, help = "sample rate in s", default = 10)
parser.add_argument("T_lisa", type = float, help = "observation time in year", default = 4)
parser.add_argument("n_start_wf", type = int, help = "start index for the parameters", default = 0)
parser.add_argument("n_end_wf", type = int, help = "end index for the parameters", default = 0)

check_memory()
# ====================== arguments parameters ==============================

args = parser.parse_args()

# if you don't pass 'n_start_wf, n_end_wf', go trough all the catalog
if args.n_end_wf == 0:
    identifier = f"{args.nmodel}_Tgw{int(args.t_gw)}_dt{int(args.delta_t)}_Tobs{int(args.T_lisa)}"
else:
    param = param[args.n_start_wf:args.n_end_wf]
    identifier = f"{args.nmodel}_Tgw{int(args.t_gw)}_{args.n_start_wf}_{args.n_end_wf}"
    
param = pd.read_hdf(f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_SNR/Model{args.nmodel}/M{identifier}_SNR_output.h5", key="paramout", index=False)
print('Number of sources:', len(param))

#take only detectable emris
param = param[param.SNR > 20].reset_index(drop=True)
print('Number of sources with SNR > 20:', len(param))

# =============== Preliminaries for response function (GPU) ================

# order of the langrangian interpolation
t0 = 20000.0   # How many samples to remove from start and end of simulations
order = 25

orbit_file_esa = "/home/mp/piarulm/work/GitHub_Repos/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"

orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

# 1st or 2nd or custom (see docs for custom)
tdi_gen = "1st generation"

index_lambda = 8
index_beta = 7

tdi_kwargs_esa = dict(
    orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AET",
    )

TDI_channels = ['TDIA','TDIE','TDIT']
N_channels = len(TDI_channels)

substring_mil    = "Need to raise max_init_len parameter"
substring_spline = "spline points"
substring_ellipt = "EllipticK failed"
substring_div_zero = "division by zero"

use_gpu = True

if use_gpu:
    xp = cp
else:
    xp = np

inspiral_kwargs = {
        "DENSE_STEPPING": 0,
        "max_init_len": int(1e8),
        "err": 1e-10,  # To be set within the class
        "use_rk8": True,
        }

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": True,  # GPU is available for this type of summation
    "pad_output": True,
}

amplitude_kwargs = {
    }

outdir = '/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_SNR/'

Path(outdir).mkdir(exist_ok=True)
waveform_model = GenerateEMRIWaveform('Pn5AAKWaveform', return_list=False, inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

### ================== Build Response Wrapper ===================
EMRI_TDI = ResponseWrapper(waveform_model, args.T_lisa, args.delta_t,
                          index_lambda,index_beta,t0=t0,
                          flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                          remove_garbage = True, **tdi_kwargs_esa)
####======================= Set Parameters and Input File =========================
if args.n_end_wf == 0: 
    param['error_fisher']  = np.zeros(len(param.M))
    
else: 
    param['error_fisher']  = np.zeros(args.n_end_wf - args.n_start_wf)
    
# Initialize Parameters
param_names = ['M', 'mu', 'a', 'p0', 'e0', 'Y0', 'dist', 'qS', 'phiS', 
               'qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0']

N_sources = len(param.M)
len_params = len(param_names)

# Initialize Storage Containers
sigmas = defaultdict(lambda: {'name': [], 'value': [], 'error': []})
fisher_matrices = []
covariance_matrices = []

# Insert true values
for j in range(N_sources):
    sigmas[j]['value'] = [param[key][j] for key in param_names]

# Determine the range of indices to process
if args.n_end_wf == 0:
    range_loop = range(len(param.M))
else:
    range_loop = range(args.n_start_wf, args.n_end_wf)

# Loop through the specified range
for j in tqdm(range_loop):
    print(j)
    print('SNR:', param.SNR[j])

    # Handle error cases for `error_fisher`
    if param.p0[j] < 0:
        param.loc[j, 'error_fisher'] = param.p0[j]
    elif param.p0[j] == 0:
        param.loc[j, 'error_fisher'] = -3
    else:
        try:
            # Compute Fisher and Covariance Matrices
            fisher_matrix, covariance_matrix = compute_fisher_matrix(j, param, args, outdir, param_names)

            if fisher_matrix is not None and covariance_matrix is not None:
                
                # Store matrices
                fisher_matrices.append(fisher_matrix)
                covariance_matrices.append(covariance_matrix)
                
                # Compute uncertainties (sigmas)
                sigma_errors = np.sqrt(np.diag(covariance_matrix))
                sigmas[j]['name'] = param_names
                sigmas[j]['error'] = sigma_errors
                
                print(f"Iteration {j}: Fisher Matrix computed and memory checked.")
                check_memory()

                del fisher_matrix, covariance_matrix

        except Exception as e:
            print(f"Error processing index {j}: {e}")
            param.loc[j, 'error_fisher'] = 4
        
        except TimeoutError:
            print(f"Skipping iteration {j} as it takes > 60 seconds)")
            param.error_fisher[j] = 1
            continue

        except ZeroDivisionError:
            logging.warning(f"Skipped source index {j} due to division by zero.")
            param.error_fisher[j] = 2
            continue  # Skip to the next source

        except ValueError as e:
            if substring_mil in str(e):
                param.error_fisher[j] = 3
                print("ValueError", e)
                        
            elif substring_spline in str(e):
                param.error_fisher[j] = 4
                print("ValueError", e)

            elif substring_ellipt in str(e):
                param.error_fisher[j] = 5
                print("ValueError", e)

# # ===================== Save the output ===================

# Convert matrices to DataFrames for storage
# Flatten Fisher and Covariance matrices
fisher_flat_list = []
covariance_flat_list = []

for j, fisher_matrix in enumerate(fisher_matrices):
    for row in range(fisher_matrix.shape[0]):
        for col in range(fisher_matrix.shape[1]):
            fisher_flat_list.append({'source': j, 'row': row, 'col': col, 'value': fisher_matrix[row, col]})

for j, cov_matrix in enumerate(covariance_matrices):
    for row in range(cov_matrix.shape[0]):
        for col in range(cov_matrix.shape[1]):
            covariance_flat_list.append({'source': j, 'row': row, 'col': col, 'value': cov_matrix[row, col]})

# Convert to DataFrames
fisher_df = pd.DataFrame(fisher_flat_list)
covariance_df = pd.DataFrame(covariance_flat_list)

# Convert uncertainties into a flattened DataFrame
errors_list = [{'idx': j, 'name': name, 'value': value, 'error': error} 
               for j, data in sigmas.items() 
               for name, value, error in zip(data['name'], data['value'], data['error'])]

errors_df = pd.DataFrame(errors_list)

# Save HDF5 Files
output_dir = f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_fisher/Model{args.nmodel}"
fisher_df.to_hdf(f"{output_dir}/M{identifier}_fisher_matrix.h5", key="fisher", index=False)
covariance_df.to_hdf(f"{output_dir}/M{identifier}_covariance_matrix.h5", key="covariance", index=False)
errors_df.to_hdf(f"{output_dir}/M{identifier}_sigma_fisher.h5", key="sigma", index=False)
paramto_hdf(f"{output_dir}/M{identifier}_paramout_fisher.h5", key="paramout", index=False)

print("Data saved successfully.")

# Violin Plot for Parameter Uncertainties (Sigma)
# LaTeX-style labels
param_latex_names = {
    'M': r'$M$',
    'mu': r'$\mu$',
    'a': r'$a$',
    'p0': r'$p_0$',
    'e0': r'$e_0$',
    'Y0': r'$Y_0$',
    'dist': r'$\mathrm{dist}$',
    'qS': r'$q_S$',
    'phiS': r'$\phi_S$',
    'qK': r'$q_K$',
    'phiK': r'$\phi_K$',
    'Phi_phi0': r'$\Phi_{\phi_0}$',
    'Phi_theta0': r'$\Phi_{\theta_0}$',
    'Phi_r0': r'$\Phi_{r_0}$'
}

# Replace names in the DataFrame with LaTeX labels
errors_df['name_latex'] = errors_df['name'].map(param_latex_names)

# Violin Plot for Parameter Uncertainties (Sigma)
plt.figure(figsize=(10, 6))
sns.violinplot(x='name_latex', y='error', data=errors_df, density_norm='width', log_scale=True, alpha=0.5, linewidth=0.8, inner=None)
plt.ylim(1e-9, 70)
plt.title(f'Parameter Uncertainties (Sigma), M{args.nmodel}')
plt.ylabel(r'$\sigma$ (Standard Deviation)')
plt.xlabel('Parameters')

# Add Medians
for i, param in enumerate(errors_df['name_latex'].unique()):
    median_val = errors_df.loc[errors_df['name_latex'] == param, 'error'].median()
    plt.plot([i-0.35, i+0.35], [median_val]*2, color='gray', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(f"{output_dir}/M{identifier}_violin.pdf")

# Violin Plot for Relative Errors
errors_df['ratio'] = errors_df['error'] / errors_df['value']
plt.figure(figsize=(10, 6))
sns.violinplot(x='name_latex', y='ratio', data=errors_df, density_norm='width', log_scale=True, alpha=0.5, linewidth=0.8, inner=None)
plt.ylim(1e-9, 70)
plt.title(f'Parameter Uncertainties (Relative Error), M{args.nmodel}')
plt.ylabel(r'$\sigma / \mathrm{value}$')
plt.xlabel('Parameters')

# Add Medians
for i, param in enumerate(errors_df['name_latex'].unique()):
    median_val = errors_df.loc[errors_df['name_latex'] == param, 'ratio'].median()
    plt.plot([i-0.35, i+0.35], [median_val]*2, color='gray', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(f"{output_dir}/M{identifier}_violin_ratio.pdf")
