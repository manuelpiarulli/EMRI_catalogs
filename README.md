# EMRI_catalogs

This is the github for the EMRI catalogs project.

Useful tools that we will be using jointly with github are:
1. Overleaf with project proposal: https://www.overleaf.com/project/65f440e2e508eeefbcb198f3
2. Environment for running the codes, by Ollie: https://github.com/OllieBurke/FastEMRIWaveforms_backwards

Inside the folder "code": 
- input.py: this code creates the input file for the trajectory code. 
- NewCat_Cal_traj.py: this code evolves backward in time the sources' parameters, it calls the function "traj_back" inside NewCat_traj.py
- EMRI_SNR.py: once we have all the initial parameters, evaluated through NewCat_Cal_traj.py, this code computes the SNR of each source. 
