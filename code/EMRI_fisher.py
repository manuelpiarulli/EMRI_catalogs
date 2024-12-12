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


parser = argparse.ArgumentParser()

parser.add_argument("nmodel", type = int, help = "model number")
parser.add_argument("delta_t", type = int, help = "sample rate in s", default = 5)
parser.add_argument("t_gw", type = int, help = "sample rate in s", default = 10)
parser.add_argument("T_lisa", type = float, help = "observation time in year", default = 4)
parser.add_argument("n_start_wf", type = int, help = "start index for the parameters", default = 0)
parser.add_argument("n_end_wf", type = int, help = "end index for the parameters", default = 0)

# ====================== arguments parameters ==============================

args = parser.parse_args()

# if you don't pass 'n_start_wf, n_end_wf', go trough all the catalog
if args.n_end_wf == 0:
    identifier = f"{args.nmodel}_Tgw{int(args.t_gw)}_dt{int(args.delta_t)}_Tobs{int(args.T_lisa)}"
else:
    param = param[args.n_start_wf:args.n_end_wf]
    identifier = f"{args.nmodel}_Tgw{int(args.t_gw)}_{args.n_start_wf}_{args.n_end_wf}"
    
param = pd.read_hdf(f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_SNR/Model{args.nmodel}/M{identifier}_SNR_output.h5", key="paramout", index=False)
print(len(param))

#take only detectable emris
param = param[param.SNR > 20].reset_index(drop=True)
print(len(param))

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
    param['fisher'] = np.zeros(len(param.M))
    param['covariance'] = np.zeros(len(param.M))
    param['error']  = np.zeros(len(param.M))
    
else: 
    param['fisher']    = np.zeros(args.n_end_wf - args.n_start_wf)
    param['covariance'] = np.zeros(args.n_end_wf - args.n_start_wf)
    param['error']  = np.zeros(args.n_end_wf - args.n_start_wf)
    

## ===================== EVALUATE THE FISHER ====================   

sigmas = []

if args.n_end_wf == 0:
    range_loop = range(len(param.M))
else: 
    range_loop = range(args.n_start_wf, args.n_end_wf)

print(f"range: {range_loop[0]} - {range_loop[-1]}")

for j in tqdm(range_loop):
    print(j)
    print('SNR:', param.SNR[j])
    if param.p0[j] < 0:
        param.error[j] = param.p0[j]
    
    elif param.p0[j] == 0: 
        param.error[j] = -3

    else:
        # try:
        true_params = [] 
        
        print(true_params)

        #varied parameters
        param_names = ['M','mu','a','p0','e0','Y0', 'dist', 'Phi_phi0','Phi_theta0','Phi_r0']

        #initialization
        fish = StableEMRIFisher(np.float64(param.M[j]*(1+param.z[j])), np.float64(param.mu[j]*(1+param.z[j])),
                        np.float64(param.a[j]), np.float64(param.p0[j]), np.float64(param.e0[j]), 
                        np.float64(param.Y0[j]), np.float64(param.dist[j]), np.float64(param.qS[j]),
                        np.float64(param.phiS[j]), np.float64(param.qK[j]), np.float64(param.phiK[j]),
                        np.float64(param.Phi_phi0[j]), np.float64(param.Phi_theta0[j]), np.float64(param.Phi_r0[j]),
                                dt=args.delta_t, T=args.T_lisa, EMRI_waveform_gen=EMRI_TDI,
                        param_names=param_names, stats_for_nerds=True, stability_plot = False, Ndelta=16,
                                use_gpu=True, filename=outdir, live_dangerously=True)

        #execution
        fisher_matrix = fish()
        covariance_matrix = np.linalg.inv(fisher_matrix)

        breakpoint()
        # param['fisher'][j] = fisher_matrix
        # param['covariance'][j] = covariance_matrix

        # # param['fisher'][j] = fish()
        # # param['covariance'][j] = np.linalg.inv(fisher_matrix)
        
        # # Compute standard deviations
        # sigma = np.sqrt(np.diag(covariancematrix))
        # sigmas.append(sigma)

        # except Exception as e:
        #     breakpoint()
        #     print(f"Error at index {j}: {e}")

        #     sigmas.append([np.nan] * len(param_names))

        # except ValueError as e:
        #     if substring_mil in str(e):
        #         param.error[j] = 1
        #         print("ValueError", e)
                        
        #     elif substring_spline in str(e):
        #         param.error[j] = 2
        #         print("ValueError", e)

        #     elif substring_ellipt in str(e):
        #         param.error[j] = 3
        #         print("ValueError", e)

# # ===================== Save the output ====================

param.to_hdf(f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_SNR/Model{args.nmodel}/M{identifier}_fisher_output.h5", key="paramout", index=False)

# # ===================== Violin Plot ====================

data = []
for i, sigma in enumerate(sigmas):
    for k, value in enumerate(sigma):
        data.append((param_names[k], value))

df = pd.DataFrame(data, columns=['Parameter', 'Sigma'])

# Create violin plot
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.violinplot(x='Parameter', y='Sigma', data=df, scale='width', log_scale=True, ax=ax1, alpha=0.5, linewidth=0.8, inner=None)
plt.ylim(1e-8, 70)
plt.title('Parameter Uncertainties (Sigma)')
plt.ylabel(r'$\sigma$ (Standard Deviation)')
plt.xlabel('Parameters')

# Add medians
for i, param in enumerate(df['Parameter'].unique()):
    median_val = df[df['Parameter'] == param]['Sigma'].median()
    plt.plot([i-0.35, i+0.35], [median_val]*2, color='gray', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_SNR/Model{args.nmodel}/M{identifier}_violin.pdf")