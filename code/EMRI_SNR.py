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
from timeout_decorator import timeout, TimeoutError
import argparse 


parser = argparse.ArgumentParser()

parser.add_argument("nmodel", type = int, help = "model number")
parser.add_argument("t_gw", type = int, help = "Tgw", default = 10)
parser.add_argument("delta_t", type = int, help = "sample rate in s", default = 20)
parser.add_argument("T_lisa", type = float, help = "observation time in year", default = 4)
parser.add_argument("n_start_wf", type = int, help = "start index for the parameters", default = 0)
parser.add_argument("n_end_wf", type = int, help = "end index for the parameters", default = 0)



# ====================== arguments parameters ==============================

args = parser.parse_args()

param = pd.read_hdf(f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_traj/Model{args.nmodel}/M{args.nmodel}_Tgw{int(args.t_gw)}_traj_tot.h5", key="paramout", index=False)
# if you don't pass 'n_start_wf, n_end_wf', go trough all the catalog
if args.n_end_wf == 0:
    identifier = f"{args.nmodel}_Tgw{int(args.t_gw)}"
else:
    param = param[args.n_start_wf:args.n_end_wf]
    identifier = f"{args.nmodel}_Tgw{int(args.t_gw)}_{args.n_start_wf}_{args.n_end_wf}"

YRSID_SI = 31558149.763545603

substring_mil    = "Need to raise max_init_len parameter"
substring_spline = "spline points"
substring_ellipt = "EllipticK failed"

use_gpu = True

if use_gpu:
    xp = cp
else:
    xp = np

def zero_pad(data):
    N = len(data)
    pow_2 = xp.ceil(np.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
    prefac = 4*args.delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)
    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj)/PSD))



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

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e8),  # all of the trajectories will be well under len = 1e6
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True,
}

##========================== Waveform Settings ========================
sampling_frequency =  1/args.delta_t   # Sampling frequency

t_max = args.T_lisa*YRSID_SI # sec
T_LISA_arr = xp.arange(0, t_max, args.delta_t)

N_t = len(zero_pad(T_LISA_arr)) # zero pad lenght

freq = xp.fft.rfftfreq(N_t, args.delta_t) # Generate fourier frequencies 
freq[0] = freq[1]   # To "retain" the zeroth frequency

# Define PSDs
freq_np = xp.asnumpy(freq)
PSD_AET = [noisepsd_AE(freq_np, includewd=args.T_lisa),noisepsd_AE(freq_np,includewd=args.T_lisa),noisepsd_T(freq_np,includewd=args.T_lisa)]
PSD_AET = [cp.asarray(item) for item in PSD_AET] # Convert to cupy array

# Construct the AAK model with 5PN trajectories

AAK_waveform_model = GenerateEMRIWaveform("Pn5AAKWaveform", inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

####======================= Set Parameters and Input File =========================
if args.n_end_wf == 0: 
    param['SNR']    = np.zeros(len(param.M))
    param['error']  = np.zeros(len(param.M))
    param['timing'] = np.zeros(len(param.M))
else: 
    param['SNR']    = np.zeros(args.n_end_wf - args.n_start_wf)
    param['error']  = np.zeros(args.n_end_wf - args.n_start_wf)
    param['timing'] = np.zeros(args.n_end_wf - args.n_start_wf)

### ================== Build Response Wrapper ===================
EMRI_TDI = ResponseWrapper(AAK_waveform_model, args.T_lisa, args.delta_t,
                          index_lambda,index_beta,t0=t0,
                          flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                          remove_garbage = True, **tdi_kwargs_esa)

## ===================== EVALUATE THE SNR ====================
def run_EMRI_TDI(true_params):
    waveform = EMRI_TDI(*true_params) 
    
    window = cp.asarray(tukey(len(waveform[0]),0)) # Window signal, reduce leakage
    EMRI_AET_w_pad = [zero_pad(window*waveform[i]) for i in range(N_channels)] # zero pad for efficiency

    EMRI_AET_fft = xp.asarray([xp.fft.rfft(item) for item in EMRI_AET_w_pad]) # Compute waveform in frequency domain

    SNR2_AET = xp.asarray([inner_prod(EMRI_AET_fft[i],EMRI_AET_fft[i],N_t,args.delta_t,PSD_AET[i]) for i in range(N_channels)])

    del waveform, EMRI_AET_fft
    return SNR2_AET, EMRI_AET_w_pad                

if args.n_end_wf == 0:
    range_loop = range(len(param.M))
else: 
    range_loop = range(args.n_start_wf, args.n_end_wf)

print(f"range: {range_loop[0]} - {range_loop[-1]}")

for j in tqdm(range_loop):
    print(j)
    
    if param.p0[j] < 0:
        param.error[j] = param.p0[j]
    
    elif param.p0[j] == 0: 
        param.error[j] = -3

    else:
        try:
            true_params = [np.float64(param.M[j]*(1+param.z[j])), np.float64(param.mu[j]*(1+param.z[j])),
                           np.float64(param.a[j]), np.float64(param.p0[j]), np.float64(param.e0[j]), 
                           np.float64(param.Y0[j]), np.float64(param.dist[j]), np.float64(param.qS[j]),
                           np.float64(param.phiS[j]), np.float64(param.qK[j]), np.float64(param.phiK[j]),
                           np.float64(param.Phi_phi0[j]), np.float64(param.Phi_theta0[j]), np.float64(param.Phi_r0[j])] 
            
            print(true_params)

            SNR2_AET, EMRI_AET_w_pad = run_EMRI_TDI(true_params)

            param.SNR[j] = cp.sum(SNR2_AET) ** (1 / 2)
            print("SNR:", param.SNR[j])
            sys.stdout.write(f"Ending processing source number: {j}\n") 

        except ValueError as e:
            if substring_mil in str(e):
                param.error[j] = 1
                print("ValueError", e)
                        
            elif substring_spline in str(e):
                param.error[j] = 2
                print("ValueError", e)

            elif substring_ellipt in str(e):
                param.error[j] = 3
                print("ValueError", e)

            # try to void memory accumulation
            del EMRI_AET_w_pad, SNR2_AET

# # ===================== Save the output ====================
param.to_hdf(f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_SNR/Model{args.nmodel}/M{identifier}_SNR_output.h5", key="paramout", index=False)

