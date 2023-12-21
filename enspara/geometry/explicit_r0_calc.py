import mdtraj as md
import yaml
import enspara
import os
import pandas as pd
import numpy as np
from numpy.linalg import norm

def load_dye(dyename, dyelibrary, dyes_dir):
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

def map_dye_on_protein(pdb, dyename, resseq, outpath, save_aligned_dyes=False, centern='', weight_dyes=False):
    '''
    Aligns a dye trajectory onto a pdb file, removing any conformations 
    that overlap with protein atoms.
    
    Attributes
    --------------
    pdb : md.Trajectory, 
        PDB of protein conformation
    dyename: string, 
        Name of dye in dye library
    resseq: int
        residue to label (using PDB ID)
    outpath: path, 
        Where to write output to
    save_aligned_dyes: bool, default=False
        optionally save trajectory of aligned/pruned dyes
    centern: int,
        protein center number that you're aligning to (for output naming)
    weights: bool, default=True
        Weight conformation probability by conformation probability in dye traj?
    
    Returns
    ---------------
    
    '''
    
    #Set the dyes directory to enspara dyes
    dyes_dir=os.path.dirname(enspara.__file__)+'/data/dyes'
    
    #Load the dyelibrary to use for parsing etc.
    with open(f'{dyes_dir}/libraries.yml','r') as yaml_file:
        dyelibrary = yaml.load(yaml_file, Loader=yaml.FullLoader)
        
    dye = load_dye(dyename, dyelibrary,dyes_dir)
  
    #Align the dye to the supplied resseq and update xyzs
    dye.xyz=dyefs.align_dye_to_res(top,dye.xyz,resseq)
    
    #Remove conformations that overlap with protein
    dye_indicies = remove_touches_protein_dye_traj(pdb, dye, resseq)
    
    #Optionally, save the aligned 
    if save_aligned_dyes:
        os.makedirs(f'{outpath}/dye-alignments',exist_ok=True)
        md.save_dcd(f'{outpath}/dye-alignments/{dyename}-center-{centern}-residue{resseq}.dcd')
        
    if weight_dyes:
        dye_weights=np.loadtxt(
            f'{dye_dir}/weights/{dyelibrary[dyename]["filename"].split("_cutoff")[0]}_cutoff10_weights.txt')
        
        dye_weights=dye_weights[dye_indicies]
        
        dye_probs = dye_weights / sum(dye_weights)
    
    return(dye_indicies)


def remove_touches_protein_dye_traj(pdb, dye, resseq):
    """
    Takes a dye trajectory and aligns it to a protein PDB structure at resseq

    
    Attributes
    --------------
    pdb : md.Trajectory, 
        PDB of protein conformation
    dye: md.Trajectory, 
        Trajectory of dye conformations
    resseq: int,
        Residue to label (using PDB ID)
    
    Returns
    ---------------
    whole_dye_indicies: np.ndarray,
        Array of dye indicies that properly map on the protein
    """
    
    #Subsection the topology to remove the replaced residue
    pdb_sliced=pdb.atom_slice(pdb.top.select(f'not resSeq {resseq}'))
    
    # Send each dye frame to check if atoms overlaps with the protein. 
    # If so, atoms are deleted. Overlap defined as any distance less than
    # the distance between the edge of the protein elemental radii 
    # + the dye elemental radii + probe radius (all in nm)
    # 0.06 approximates a H-bond.
    # This returns a list of atoms that are not touching protein
    atoms_not_touching_protein=np.array(
        [np.shape(
            dyefs.remove_touches_protein(i, pdb_sliced, probe_radius=0.06))[0] 
         for i in dye.xyz])
    
    #Select out the whole dyes, with a slight tolerance for backbone atom overlaps
    whole_dye_indicies=np.where(
        atoms_not_touching_protein>=len(dye.xyz[0])-6)[0]
    
    return whole_dye_indicies