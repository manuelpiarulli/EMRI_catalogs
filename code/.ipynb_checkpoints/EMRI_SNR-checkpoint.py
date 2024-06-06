import numpy as np
import cupy as cp
import pandas as pd
import time
import h5py
from scipy.signal import tukey
from lisatools.sensitivity import noisepsd_AE,noisepsd_T
from lisatools.diagnostic import *
import sys
from tqdm import tqdm as tqdm
from few.waveform import GenerateEMRIWaveform, Pn5AAKWaveform
from fastlisaresponse import ResponseWrapper
from timeout_decorator import timeout, TimeoutError
import argparse 

# === BEFORE RUNNING THIS CODE YOU NEED TO RUN traj_manuel.py, TO SOLVE TRAJECTORIES ISSUES ===

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help = "Input file as: param_input_model8_cut_SNR_1.h5 for model 8")
parser.add_argument("folder", help = "Reference folder for input and results")
parser.add_argument("nmodel", type = int, help = "model number")
parser.add_argument("delta_t", type = int, help = "sample rate in s", default = 20)
parser.add_argument("T", type = float, help = "observation time in year", default = 4)
parser.add_argument("n_start_wf", type = int, help = "start index for the parameters", default = None)
parser.add_argument("n_end_wf", type = int, help = "end index for the parameters", default = None)
parser.add_argument("snr_threshold", type = int, help = "snr threshold", default = 20)
parser.add_argument("cut_snr", type = int, help = "cut snr in the catalog", default = None)

# ====================== arguments parameters ==============================

args = parser.parse_args()

param = pd.read_hdf(args.input_file, key = "param")

param = param[args.n_start_wf:args.n_end_wf]

# if you don't pass 'n_start_wf, n_end_wf', go trough all the catalog
if args.n_start_wf == None:
    args.n_start_wf = 0
    args.n_end_wf   = len(param.M)

# cut_snr argument
if args.cut_snr == None:
    identifier = f"{args.nmodel}_{args.n_start_wf}_{args.n_end_wf}"
elif args.cut_snr == 1:
    identifier = f"{args.nmodel}_cut_SNR_{args.cut_snr}_{args.n_start_wf}_{args.n_end_wf}"
elif args.cut_snr == 101:
    identifier = f"{args.nmodel}_cut_SNR_01_{args.n_start_wf}_{args.n_end_wf}"


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
    "max_init_len": int(1e8),  # all of the trajectories will be well under len = 100000
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True,
}

##========================== Waveform Settings ========================
#args.delta_t = 20        # Sampling interval in seconds
#args.T = 4.0         # Time elapsed for signal in years
sampling_frequency =  1/args.delta_t   # Sampling frequency

# Construct the AAK model with 5PN trajectories

AAK_waveform_model = GenerateEMRIWaveform("Pn5AAKWaveform", inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)
#AAK_waveform_model = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

####======================= Set Parameters and Input File =========================
param['SNR']    = np.zeros(args.n_end_wf - args.n_start_wf)
param['error']  = np.zeros(args.n_end_wf - args.n_start_wf)
param['timing'] = np.zeros(args.n_end_wf - args.n_start_wf)

year  = 31558149.763545603 #sec
t_max = args.T*year # sec
T_LISA_arr = xp.arange(0, t_max, args.delta_t)

N_t = len(zero_pad(T_LISA_arr)) # zero pad lenght

# unresolvable sources              
TDI_tot_A_u = xp.zeros(N_t)
TDI_tot_E_u = xp.zeros(N_t)
TDI_tot_T_u = xp.zeros(N_t)

# resolvable sources
TDI_tot_A_r = xp.zeros(N_t)
TDI_tot_E_r = xp.zeros(N_t)
TDI_tot_T_r = xp.zeros(N_t)

freq = xp.fft.rfftfreq(N_t,args.delta_t) # Generate fourier frequencies 
freq[0] = freq[1]   # To "retain" the zeroth frequency

# Define PSDs
freq_np = xp.asnumpy(freq)
PSD_AET = [noisepsd_AE(freq_np, includewd=args.T),noisepsd_AE(freq_np,includewd=args.T),noisepsd_T(freq_np,includewd=args.T)]
PSD_AET = [cp.asarray(item) for item in PSD_AET] # Convert to cupy array

### ================== Build Response Wrapper ===================
EMRI_TDI = ResponseWrapper(AAK_waveform_model,args.T,args.delta_t, index_lambda,index_beta,t0=t0,
                          flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                          remove_garbage = True, **tdi_kwargs_esa)

## ===================== EVALUATE THE GWB ====================
@timeout(60)  # Imposta il timeout a 60 secondi
def run_EMRI_TDI(true_params):
    waveform = EMRI_TDI(*true_params) 
    
    window = cp.asarray(tukey(len(waveform[0]),0)) # Window signal, reduce leakage
    EMRI_AET_w_pad = [zero_pad(window*waveform[i]) for i in range(N_channels)] # zero pad for efficiency
    EMRI_AET_fft = xp.asarray([xp.fft.rfft(item) for item in EMRI_AET_w_pad]) # Compute waveform in frequency domain
    SNR2_AET = xp.asarray([inner_prod(EMRI_AET_fft[i],EMRI_AET_fft[i],N_t,args.delta_t,PSD_AET[i]) for i in range(N_channels)])

    del waveform, EMRI_AET_fft
    return SNR2_AET, EMRI_AET_w_pad

for j in tqdm(range(args.n_start_wf, args.n_end_wf)):
    try:
        timei = time.time()
        true_params = [np.float64(param.M[j]), np.float64(param.mu[j]), np.float64(param.a[j]), np.float64(param.p0[j]), 
        np.float64(param.e0[j]), np.float64(param.Y0[j]), np.float64(param.dist[j]), np.float64(param.qS[j]), np.float64(param.phiS[j]),
        np.float64(param.qK[j]), np.float64(param.phiK[j]), np.float64(param.Phi_phi0[j]), np.float64(param.Phi_theta0[j]),
        np.float64(param.Phi_r0[j])]

        SNR2_AET, EMRI_AET_w_pad = run_EMRI_TDI(true_params)

        param.SNR[j] = cp.sum(SNR2_AET) ** (1 / 2)
        print("SNR:", param.SNR[j])

        sys.stdout.write(f"Ending processing source number: {j}\n") 
        param.timing[j] = time.time() - timei 
        
        if param.SNR[j] < args.snr_threshold:
            print("gwb source")
            TDI_tot_A_u += cp.asarray(EMRI_AET_w_pad[0])
            TDI_tot_E_u += cp.asarray(EMRI_AET_w_pad[1])
            TDI_tot_T_u += cp.asarray(EMRI_AET_w_pad[2])
        
        else:                              
            print('resolvable source')
            TDI_tot_A_r += cp.asarray(EMRI_AET_w_pad[0])
            TDI_tot_E_r += cp.asarray(EMRI_AET_w_pad[1])
            TDI_tot_T_r += cp.asarray(EMRI_AET_w_pad[2])

    except TimeoutError:
        print(f"Skipping iteration {j} as it takes > 60 seconds)")
        param.SNR[j] = 0
        param.error[j] = 4
        param.timing[j] = 40
        continue
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

t_new = np.arange(0,len(TDI_tot_A_u)*args.delta_t,args.delta_t)

if use_gpu:
    dictio_TDI_u = {'t' : t_new,
                    'TDI_tot_A_u': xp.asnumpy(TDI_tot_A_u),
                    'TDI_tot_E_u': xp.asnumpy(TDI_tot_E_u),
                    'TDI_tot_T_u': xp.asnumpy(TDI_tot_T_u)} 
    # resolvable sources
    dictio_TDI_r = {'t' : t_new,
                    'TDI_tot_A_r': xp.asnumpy(TDI_tot_A_r),
                    'TDI_tot_E_r': xp.asnumpy(TDI_tot_E_r),
                    'TDI_tot_T_r': xp.asnumpy(TDI_tot_T_r)} 
else:
    dictio_TDI_u = {'t' : t_new,
                    'TDI_tot_A_u': TDI_tot_A_u,
                    'TDI_tot_E_u': TDI_tot_E_u,
                    'TDI_tot_T_u': TDI_tot_T_u}
    # resolvable sources
    dictio_TDI_r = {'t' : t_new,
                    'TDI_tot_A_r': TDI_tot_A_r,
                    'TDI_tot_E_r': TDI_tot_E_r,
                    'TDI_tot_T_r': TDI_tot_T_r}

# save LISA_PSD
# with h5py.File(f"{args.folder}/PSD_AET_model{args.nmodel}.h5", "w") as hf:
#    for i, arr in enumerate(PSD_AET):
#        hf.create_dataset(f'PSD_AET_{i}', data=cp.asnumpy(arr))

# # ===================== Save the output ====================

param.to_hdf(f"{args.folder}/param_output_model{identifier}.h5", key="paramout", index=False)
#unresolvable sources (gwb) 
TDI_dictio_u = pd.DataFrame(dictio_TDI_u)
TDI_dictio_u.to_hdf(f"{args.folder}/TDI_sum_u_model{identifier}.h5", key = "TDI_sum_u", index = False)
#resolvable sources
TDI_dictio_r = pd.DataFrame(dictio_TDI_r)
TDI_dictio_r.to_hdf(f"{args.folder}/TDI_sum_r_model{identifier}.h5", key = "TDI_sum_r", index = False)  
