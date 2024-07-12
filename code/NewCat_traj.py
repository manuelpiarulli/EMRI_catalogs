import numpy as np
from few.utils.utility import (get_separatrix, )
#######====================Traj====================
from few.trajectory.inspiral import EMRIInspiral
#from simplified_waveform import T, LISA_yr
import os

T=1*365.25*24*60*60
LISA_yr = 4*T

trajectory_class = 'pn5'
print("5PN AAK trajectory")

inspiral_kwargs_traj = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e6)# all of the trajectories will be well under len = 1e6
}

traj_backwards = EMRIInspiral(func = trajectory_class, integrate_backwards = True) 

def traj_back(nmodel,i, M, mu, a, e_f, x_f, Y_f, Tgw, Phi_phi0, Phi_theta0, Phi_r0, pf_init = 0.30):
    if mu/M > 1e-3: #mass ratio limit
        Y0, e0, p0, Y4, e4, p4  = 0, 0, 0, 0, 0, 0
    else:
        try:
            p_f = get_separatrix(a, e_f, x_f) + pf_init   

            print("For source {}".format(i))
            print("mass ratio of the system is: {}".format(np.round(mu/M,8)))
            print("Final eccentricity = {}".format(e_f))
            print("Final semi-latus rectum = {}".format(p_f))
            print("The separatrix should be = {}".format(p_f - pf_init))

            (t_back, p_back, 
            e_back, Y_back, 
            Phi_phi_back, 
            Phi_r_back, 
            Phi_theta_back) = traj_backwards(M, 
                                            mu, a, p_f, e_f, Y_f, 
                                            Phi_phi0=Phi_phi0, 
                                            Phi_theta0=Phi_theta0, 
                                            Phi_r0=Phi_r0, 
                                            T=Tgw, 
                                            **inspiral_kwargs_traj)

            Y0, e0, p0 = Y_back[-1], e_back[-1], p_back[-1]
        except:
                print(i, "Round 1 Except Wrong!!!")
                try:
                    p_f = get_separatrix(a, e_f, x_f) + pf_init
                    (t_back, p_back, 
                    e_back, Y_back, 
                    Phi_phi_back, 
                    Phi_r_back, 
                    Phi_theta_back) = traj_backwards(M, 
                                                    mu, a, p_f, e_f, Y_f, 
                                                    Phi_phi0=Phi_phi0, 
                                                    Phi_theta0=Phi_theta0, 
                                                    Phi_r0=Phi_r0, 
                                                    T=Tgw, 
                                                    **inspiral_kwargs_traj)
                    Y0, e0, p0 = Y_back[-1], e_back[-1], p_back[-1]
                except:
                    print(i, "Round 2 Except Wrong!!! ")
                    print(f"redshifted mass: {M, mu}, spin: {a}, pf: {p_f}, ef: {e_f}, Yf: {Y_f}, Tgw, {Tgw}")
                    print('\n')
                    Y0, e0, p0, Y4, e4, p4 = 0, 0, -2, 0, 0, 0
        if p0 > 0: #integrate back success
            time_ratio = (Tgw)/(t_back[-1]/T)
            if  time_ratio > 1.1 or time_ratio < 0.9:
                print(i, "Time Wrong!!!")
                print(f"redshifted mass: {M, mu}, spin: {a}, pf: {p_f}, ef: {e_f}, Yf: {Y_f}, Tgw, {Tgw}")
                print(f"e0: {e_back[-1]}, Y0: {Y_back[-1]}, p0: {p_back[-1]}") 
                print("This two should be the same:", Tgw, t_back[-1]/T, '\n')
                Y0, e0, p0, Y4, e4, p4  = 0, 0, -1, 0, 0, 0
            else:
                try:
                    if Tgw > LISA_yr:
                        point = len((t_back/T)[(t_back/T)<(Tgw-LISA_yr)])
                        t4 = t_back[point]
                        p4 = p_back[point]
                        e4 = e_back[point]
                        Y4 = Y_back[point]
                    else:
                        p4 = p_f
                        e4 = e_f
                        Y4 = Y_f
                except:
                    print(i, "T4 wrong", '\n')

#    np.savetxt(file_path, np.array([Y0, e0, p0, Y4, e4, p4]))
    return  Y0, e0, p0, Y4, e4, p4
