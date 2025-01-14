#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from astropy.cosmology import FlatLambdaCDM
#from NewCat_Source import NewCat_Source
import astropy.units as u
import argparse
import copy
from tqdm import tqdm 
import argparse 

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

parser = argparse.ArgumentParser()
parser.add_argument("nmodel", type = int, help = "model number")
parser.add_argument("t_gw", type = float, help = "Tgw", default = 10)


args = parser.parse_args()

def NewCat_Source(model, t_gw):
    dat = np.genfromtxt( f"/work/LISA/piarulm/EMRI_catalogs/Stas_Model/Model{model}/all_EMRI.dat")
    logM, redshift, spin, inc, dl_Gpc = [dat.T[i] for i in range(5)] #list(zip(dat.T))

    dat = np.genfromtxt(f"/work/LISA/piarulm/EMRI_catalogs/Stas_Model/Model{model}/all_EMRI_spin.dat")
    logM, redshift, unkown, R_LSO = [dat.T[i] for i in range(4)]
    N_total = np.shape(dat)[0]
    
    t_gw = int(t_gw)
    
    # rescale the number of sources depending on t_gw
    N_sources = int(N_total * t_gw/10)
    
    print("Model M", model, ", Total number in 10 yrs:", N_total)
    print("Model M", model, f", Total number in {t_gw} yrs:", N_sources)
        
    # LVK mass function for m2
    m2_data = pd.read_csv( f"/work/LISA/piarulm/EMRI_catalogs/m2_samples.txt", header=None)
    # Estrai N_sources righe
    m2 = m2_data.sample(N_sources)
    
    if t_gw != 10: 
        
        #Generate random indices for sampling
        random_indices = np.random.choice(N_total, size=N_sources, replace=False)

        #Use these indices to sample from both arrays
        M = 10**(logM[random_indices])
        mu = np.asarray(m2)
        mu = m2[0].values
        a = spin[random_indices]
        z = redshift[random_indices]
    
        print('Sources that need change inc:', len(inc[(inc>87/90*np.pi/2) & (inc<93/90*np.pi/2)]), '\n')
        #do that ALSO after the integration
        inc[(inc>87/90*np.pi/2) & (inc<90/90*np.pi/2)] = 87/90*np.pi/2
        inc[(inc>=90/90*np.pi/2) & (inc<93/90*np.pi/2)] = 93/90*np.pi/2
        inc_f = inc[random_indices]
        
        R_LSO = R_LSO[random_indices]
        
        
    else: 
        M = 10**(logM)
        mu = np.asarray(m2)
        mu = m2[0].values
        a = spin
        z = redshift
        
        print('Sources that need change inc:', len(inc[(inc>87/90*np.pi/2) & (inc<93/90*np.pi/2)]), '\n')
        #do that ALSO after the integration
        inc[(inc>87/90*np.pi/2) & (inc<90/90*np.pi/2)] = 87/90*np.pi/2
        inc[(inc>=90/90*np.pi/2) & (inc<93/90*np.pi/2)] = 93/90*np.pi/2
        inc_f = inc

    np.random.seed(27*model)
    phiS=np.random.uniform(0.,2*pi, N_sources)
    cos_qS=np.random.uniform(-1.0,1., N_sources)
    phiK=np.random.uniform(0.,2*pi, N_sources)
    cos_qK=np.random.uniform(-1.,1., N_sources)
    qS = np.arccos(cos_qS)
    qK = np.arccos(cos_qK)
    Phi_phi0 = np.random.uniform(0.,2*pi, N_sources)
    Phi_theta0 = np.random.uniform(0.,2*pi, N_sources)
    Phi_r0 = np.random.uniform(0.,2*pi, N_sources)
    x_f = np.cos(inc_f)
    Y_f = x_f #assume Y==x now
    e_f = np.random.uniform(0, 0.2, N_sources)
    Tgw = np.random.uniform(0, t_gw, N_sources) 

    return M, mu, R_LSO, z, e_f, Tgw, a, inc_f, x_f, Y_f, phiS, cos_qS, phiK, cos_qK, qS, qK, Phi_phi0, Phi_theta0, Phi_r0

pi=np.pi
data = np.asarray(NewCat_Source(args.nmodel, args.t_gw))
M, mu, R_LSO, z, e_f, Tgw, a, inc_f, x_f, Y_f, phiS, cos_qS, phiK, cos_qK, qS, qK, Phi_phi0, Phi_theta0, Phi_r0 = data

dist = cosmo.luminosity_distance(z).value/1000 #Gpc Luminosity Distance 

kwargs = {'M': M,
         'mu':         mu,
         'a':          a,
         'qS':         qS,
         'phiS':       phiS,
         'qK':         qK, 
         'phiK':       phiK,  
         'dist':       dist, 
         'Phi_phi0':   Phi_phi0,
         'Phi_theta0': Phi_theta0,
         'Phi_r0':     Phi_r0, 
         'z':          z, 
         'dist':       dist,
         'R_LSO':      R_LSO,
          'e_f':       e_f,
          'Tgw':       Tgw,
          'inc_f':     inc_f,
          'x_f':       x_f,
          'Y_f':       Y_f
                 }

# df = pd.DataFrame(kwargs)
# df.to_hdf(f"/work/LISA/piarulm/EMRI_catalogs/New_Catalog/AAK_traj/Model{args.nmodel}/M{args.nmodel}_Tgw{int(args.t_gw)}_input.h5", key = "param", index = False)
