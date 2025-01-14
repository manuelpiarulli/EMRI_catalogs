import numpy as np

import sys
sys.path.append('/work/LISA/piarulm/EMRI_catalogs/SNR_selection')

from NewCat_Source import NewCat_Source
from NewCat_traj import traj_back
import multiprocessing as mp
from tqdm import tqdm as tqdm
import os
import argparse 
import pandas as pd 

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from timeout_decorator import timeout, TimeoutError

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

parser = argparse.ArgumentParser()

parser.add_argument("nmodel", type = int, help = "model number")
parser.add_argument("t_gw", type = int, help = "Tgw", default = 10)
parser.add_argument("n_start_wf", type = int, help = "start index for the parameters", default = 0)
parser.add_argument("n_end_wf", type = int, help = "end index for the parameters", default = 0)

@timeout(60)
def traj_function(nmodel, j, param): 
    Y0, e0, p0, Y4, e4, p4 = traj_back(nmodel,j, np.float64(param.M[j]*(1+param.z[j])), np.float64(param.mu[j]*(1+param.z[j])), np.float64(param.a[j]), np.float64(param.e_f[j]), 
                                        np.float64(param.x_f[j]), np.float64(param.Y_f[j]), np.float64(param.Tgw[j]), 
                                        np.float64(param.Phi_phi0[j]), np.float64(param.Phi_theta0[j]), np.float64(param.Phi_r0[j]))
    return Y0, e0, p0, Y4, e4, p4

# ====================== arguments parameters ==============================

args = parser.parse_args()

param = pd.read_hdf(f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_traj/Model{args.nmodel}/M{args.nmodel}_Tgw{int(args.t_gw)}_input.h5", key = "param")

# if you don't pass 'n_start_wf, n_end_wf', go trough all the catalog
if args.n_end_wf == 0:
    identifier = f"{args.nmodel}_Tgw{int(args.t_gw)}"
else:
    param = param[args.n_start_wf:args.n_end_wf]
    identifier = f"{args.nmodel}_Tgw{int(args.t_gw)}_{args.n_start_wf}_{args.n_end_wf}"

GPU = False
if GPU:
    import cupy as cp
    xp = cp
else:
    xp = np

####======================= Set Parameters and Input File =========================


if args.n_end_wf == 0: 
    param['Y0'] = np.zeros(len(param.M))
    param['e0'] = np.zeros(len(param.M))
    param['p0'] = np.zeros(len(param.M))
    param['Y4'] = np.zeros(len(param.M))
    param['e4'] = np.zeros(len(param.M))
    param['p4'] = np.zeros(len(param.M))
else: 
    param['Y0'] = np.zeros(args.n_end_wf - args.n_start_wf)
    param['e0'] = np.zeros(args.n_end_wf - args.n_start_wf)
    param['p0'] = np.zeros(args.n_end_wf - args.n_start_wf)
    param['Y4'] = np.zeros(args.n_end_wf - args.n_start_wf)
    param['e4'] = np.zeros(args.n_end_wf - args.n_start_wf)
    param['p4'] = np.zeros(args.n_end_wf - args.n_start_wf)    


print(f"q < 1e3: {len(param.M[param.M/param.mu<1e3])}")

if args.n_end_wf == 0:
    range = range(len(param.M))
else: 
    range = range(args.n_start_wf, args.n_end_wf)

print(f"range: {range[0]} - {range[-1]}")

for j in tqdm(range):

    try:
        Y0, e0, p0, Y4, e4, p4  = traj_function(args.nmodel,j, param)
        
        # traj_back(args.nmodel,j, np.float64(param.M[j]*(1+param.z[j])), np.float64(param.mu[j]*(1+param.z[j])), np.float64(param.a[j]), np.float64(param.e_f[j]), 
        #                                     np.float64(param.x_f[j]), np.float64(param.Y_f[j]), np.float64(param.Tgw[j]), 
        #                                     np.float64(param.Phi_phi0[j]), np.float64(param.Phi_theta0[j]), np.float64(param.Phi_r0[j]))
        
        param.loc[j, ["Y0", "e0", "p0", "Y4", "e4", "p4"]] = [Y0, e0, p0, Y4, e4, p4]
        print(Y0, e0, p0, Y4, e4, p4)
    except TimeoutError:
            print(f"Skipping iteration {j} as it takes > 60 seconds)")
            continue
    
# Constants
# theta1 = 87 / 90 * np.pi / 2
# theta2 = 90 / 90 * np.pi / 2
# theta3 = 93 / 90 * np.pi / 2

# cos_theta1 = np.cos(theta1)
# cos_theta2 = np.cos(theta2)
# cos_theta3 = np.cos(theta3)

# # Update the Y0 vector based on the conditions
# param['Y0'][(param['Y0'] < cos_theta1) & (param['Y0'] > cos_theta2)] = cos_theta1
# param['Y0'][(param['Y0'] <= cos_theta2) & (param['Y0'] > cos_theta3)] = cos_theta3

param.Y0[(param.Y0<0.04)&(param.Y0>-0.04)]= np.random.uniform(-1, -0.2, len(param.Y0[(param.Y0<0.04)&(param.Y0>-0.04)]))

# Stampa la lunghezza degli elementi di Y0 che soddisfano la condizione
# print(len(param['Y0'][(param['Y0'] < cos_theta1) & (param['Y0'] > cos_theta3)]))


param.to_hdf(f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_traj/Model{args.nmodel}/M{identifier}_traj_output.h5", key="paramout", index=False)
print("end of run" )
         

