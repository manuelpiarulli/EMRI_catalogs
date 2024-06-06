import numpy as np
from NewCat_Source import NewCat_Source
from NewCat_traj import traj_back
import multiprocessing as mp
from tqdm import tqdm as tqdm
import os
import argparse 
import pandas as pd 

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

parser = argparse.ArgumentParser()

parser.add_argument("nmodel", type = int, help = "model number")
parser.add_argument("t_gw", type = int, help = "Tgw", default = 10)

# ====================== arguments parameters ==============================

args = parser.parse_args()

param = pd.read_hdf(f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_traj/Model{args.nmodel}/M{args.nmodel}_Tgw{int(args.t_gw)}_input.h5", key = "param")

GPU = False
if GPU:
    import cupy as cp
    xp = cp
else:
    xp = np

N_tot = len(param['M'])

####======================= Set Parameters and Input File =========================
param['Y0'] = np.zeros(N_tot)
param['e0'] = np.zeros(N_tot)
param['p0'] = np.zeros(N_tot)
param['Y4'] = np.zeros(N_tot)
param['e4'] = np.zeros(N_tot)
param['p4'] = np.zeros(N_tot)

print(f"q < 1e3: {len(param.M[param.M/param.mu<1e3])}")

for j in tqdm(range(N_tot)):
    Y0, e0, p0, Y4, e4, p4  = traj_back(args.nmodel,j, np.float64(param.M[j]*(1+param.z[j])), np.float64(param.mu[j]*(1+param.z[j])), np.float64(param.a[j]), np.float64(param.e_f[j]), 
                                        np.float64(param.x_f[j]), np.float64(param.Y_f[j]), np.float64(param.Tgw[j]), 
                                        np.float64(param.Phi_phi0[j]), np.float64(param.Phi_theta0[j]), np.float64(param.Phi_r0[j]))
    
    param.loc[j, ["Y0", "e0", "p0", "Y4", "e4", "p4"]] = [Y0, e0, p0, Y4, e4, p4]
    print(Y0, e0, p0, Y4, e4, p4)

param.to_hdf(f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_traj/Model{args.nmodel}/M{args.nmodel}_Tgw{int(args.t_gw)}_traj_output.h5.h5", key="paramout", index=False)
print("end of run" )
         

