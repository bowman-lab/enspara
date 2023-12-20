import mdtraj as md
import yaml
import enspara
import os.path
import pandas as pd
import numpy as np
from numpy.linalg import norm

#Take input of dye name and the library
#Consider adding flow control for user to add own info without adding dyes?

def load_dye(dyename, dyelibrary):
    #Set the dyes directory to enspara dyes (unless user provides)
    dyes_dir=os.path.dirname(enspara.__file__)+'/data/dyes'
    dye_file=dyelibrary[dyename]["filename"].split("_cutoff")[0]

    #Load the dye and dye weights
    dye=md.load(dyes_dir+f'/trajs/{dye_file}_cutoff10.dcd',top=dyes_dir+f'/structures/{dye_file}.pdb')
    return(dye)
    
def get_dipole_components(dye, dyename, dyelibrary):
    '''
    Takes input of a dye name in the enspara directory, loads the dye traj,
    pulls the dipole atoms, and returns the dipole moments for all frames.
    '''

    #Pull the atom IDs that comprise the dipole moment
    mu_atomids=dye.topology.select(
        f'(name {dyelibrary[dyename]["mu"][0]}) or (name {dyelibrary[dyename]["mu"][1]})')

    #Select the dipole atoms from the trajectory
    mu_positions=dye.atom_slice(mu_atomids).xyz

    #Make the dipole vector
    mu_vectors=np.subtract(mu_positions[:,0,:],mu_positions[:,1,:])
    return(mu_positions[:,0,:], mu_vectors)

def calc_R0(k2, QD, J, n=1.333):
    #n=1.333 is refractive index for water
    R0constants= 0.02108 #for R0 in nm
    n4=n**4
    return(R0constants * np.power(k2 * QD * J / n4, 1 / 6))

def get_dye_overlap(donorname, acceptorname, dyelibrary):
    dyes_dir=os.path.dirname(enspara.__file__)+'/data/dyes'
    donor_fluor=donorname.split(" ")[0]
    donor_number=donorname.split(" ")[1]
    acceptor_fluor=acceptorname.split(" ")[0]
    acceptor_number=acceptorname.split(" ")[1]
    
    
    #Load donor dye spectrum
    donor_spectrum = pd.read_csv(f'{dyes_dir}/R0/{donor_fluor}{donor_number}.csv')
    donor_spectrum[['Emission', 'Excitation']] = donor_spectrum[['Emission', 'Excitation']] / 100
    
    #Load acceptor dye spectrum
    acceptor_spectrum = pd.read_csv(f'{dyes_dir}/R0/{acceptor_fluor}{acceptor_number}.csv')
    acceptor_spectrum[['Emission', 'Excitation']] = acceptor_spectrum[['Emission', 'Excitation']] / 100
    
    #Load chromophore data
    chromophore_data = pd.read_csv(f'{dyes_dir}/R0/Dyes_extinction_QD.csv',delimiter=',',
                                   names=['Type', 'Chromophore', 'Ext_coeff', 'QD'])
    
    #Pull Quantum yield of the donor absent acceptor
    QD = chromophore_data['QD'].loc[(chromophore_data['Chromophore'] == donor_number) &
                            (chromophore_data['Type'] == donor_fluor)].values.astype(float)
    
    #Pull max extinction coefficient for the acceptor
    ext_coeff_max = chromophore_data['Ext_coeff'].loc[(chromophore_data['Chromophore'] == acceptor_number) &
                            (chromophore_data['Type'] == acceptor_fluor)].values.astype(float)
    
   # Extinction coefficient spectrum of the acceptor
    ext_coeff_acceptor = (ext_coeff_max * acceptor_spectrum['Excitation']).fillna(0)

    # Integral of the donor emission spectrum
    donor_spectra_integral = np.trapz(donor_spectrum['Emission'], x=donor_spectrum['Wavelength'])
    
    # Overlap integral between donor-acceptor (normalized by the donor emission spectrum)
    J = np.trapz(donor_spectrum['Emission'] * ext_coeff_acceptor * donor_spectrum['Wavelength'] ** 4,
             x=donor_spectrum['Wavelength']) / donor_spectra_integral
    
    return(J, QD)