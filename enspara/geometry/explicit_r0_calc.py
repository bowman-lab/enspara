import mdtraj as md
import yaml
import enspara
import os
import pandas as pd
import numpy as np
import scipy
from numpy.linalg import norm
from enspara.geometry import dyes_from_expt_dist as dyefs
from functools import partial
from multiprocessing import Pool

def load_dye(dyename, dyelibrary, dyes_dir):
    """
    Helper function for loading dyes from the enspara dye library.
    """
    
    dye_file=dyelibrary[dyename]["filename"].split("_cutoff")[0]

    #Load the dye and dye weights
    dye=md.load(dyes_dir+f'/trajs/{dye_file}_cutoff10.dcd',top=dyes_dir+f'/structures/{dye_file}.pdb')
    return(dye)

def calc_R0(k2, QD, J, n=1.333):
    """
    Calculates R0 from dye parameters
    
    Attributes:
    ------------
    k2 : float,
        Value of kappa squared
    QD : float,
        Quantum yield of donor dye
    J : float,
        Normalized Spectral overlap integral of donor
        and acceptor dye-pairs
    n : float, default = 1.333
    refractive index. Defaults to water.
    
    Returns:
    ----------
    R0 : float,
        Value for R0 in nm.
    """
    R0constants= 0.02108 #for R0 in nm
    n4=n**4
    return(R0constants * np.power(k2 * QD * J / n4, 1 / 6))

def get_dye_overlap(donorname, acceptorname):
    """
    Calculates dye parameters for calculating R0
    
    Attributes
    -------------
    donorname : string,
        name of donor dye found in enspara's dye library
    acceptorname : string,
        name of acceptor dye found in enspara's dye library   

    Returns
    -------------
    J : float,
        Normalized spectral overlap integral
    QD : float,
        Quantum yield of the donor dye
    """
    
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

def remove_touches_protein_dye_traj(pdb, dye, resseq, probe_radius=0.06):
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
    probe_radius: float,
        radius of a probe to see whether residues are overlapping in nm.
    
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
            dyefs.remove_touches_protein(i, pdb_sliced, probe_radius=probe_radius))[0] 
         for i in dye.xyz])
    
    #Select out the whole dyes, with a slight tolerance for backbone atom overlaps
    whole_dye_indicies=np.where(
        atoms_not_touching_protein>=len(dye.xyz[0])-6)[0]
    
    return whole_dye_indicies
    
    
def get_dipole_components(dye, dyename, dyelibrary):
    '''
    Takes input of a dye trajectory that exists in the the enspara library,
    pulls the dipole atoms, and returns the dipole moments for all frames.
    '''

    #Pull the atom IDs that comprise the dipole moment
    mu_atomids=dye.topology.select(
        f'(name {dyelibrary[dyename]["mu"][0]}) or (name {dyelibrary[dyename]["mu"][1]})')

    #Select the dipole atoms from the trajectory
    mu_positions=dye.atom_slice(mu_atomids).xyz

    #Make the dipole vector
    mu_vectors=np.subtract(mu_positions[:,0,:],mu_positions[:,1,:])
    
    #Return the dipole origin and the dipole vector (not unit vector!)
    return(mu_positions[:,0,:], mu_vectors)

def get_dye_center(dye, dyename, dyelibrary):
    '''
    Takes input of a dye trajectory that exists in the the enspara library,
    pulls the flurophore center position, and returns it for all frames.
    '''
    #Pull the atom IDs that comprise the dipole moment
    r_atomids=dye.topology.select(
        f'(name {dyelibrary[dyename]["r"][0]})')

    #Select the dipole atoms from the trajectory
    r_positions=dye.atom_slice(r_atomids).xyz
    
    return(r_positions.reshape((-1,3)))

def assemble_dye_r_mu(dye, dyename, dyelibrary):
    '''
    Takes input of a dye trajectory that exists in the the enspara library,
    exracts dye emission/excitation center and dipole moment for each frame in traj.
    Assembles output to bundle as a h5 file for future use.
    
    Returns
    dye_pos_params, nd.array, shape=(n_frames,6)
    First 3 positions are the xyz of the dye_center
    Second 3 give the unit vector of the dipole moment
    '''
    
    dye_center_coords=get_dye_center(dye, dyename, dyelibrary)
    
    dipole_origin, dipole_vector = get_dipole_components(dye, dyename, dyelibrary)
    
    dye_pos_params=np.hstack((dye_center_coords,dipole_origin, dipole_vector))
    return(dye_pos_params)

def calc_k2(Donor_coords, Acceptor_coords):
    """
    Calculates k2 from acceptor and donor dye positions/vectors
    
    Attributes
    --------------
    Donor_coords, nd.array (9,)
        numpy array specifying the xyz of the dye emission/excitation center,
        the origin of the dipole moment, and the dipole vector
    Acceptor_coords, nd.array (9,)
        numpy array specifying the xyz of the dye emission/excitation center,
        the origin of the dipole moment, and the dipole vector
    
    Returns
    --------------
    k2 : float,
        kappa squared value for the specified donor/acceptor positions
    """
    
    D_center, D_dip_ori, D_vec = np.split(Donor_coords, 3)
    A_center, A_dip_ori, A_vec = np.split(Acceptor_coords, 3)

    #Calculate the distance between dye emission/excitation centers
    r=scipy.spatial.distance.cdist(D_center.reshape(1,3), A_center.reshape(1,3))[0,0]

    #Define the vector joining donor and acceptor origins
    rvec=np.subtract(D_dip_ori,A_dip_ori)

    #Calculate the angles between dipole vectors
    cos_theta_T=np.dot(A_vec,D_vec)/(norm(A_vec)*norm(D_vec))
    cos_theta_D=np.dot(rvec,D_vec)/(norm(rvec)*norm(D_vec))
    cos_theta_A=np.dot(A_vec,rvec)/(norm(A_vec)*norm(rvec))

    #Calculate k2
    k2=(cos_theta_T-(3*cos_theta_D*cos_theta_A))**2
    return(k2)

def _map_dye_on_protein(pdb, dye, resseq, dyename, 
    outpath='.', save_aligned_dyes=False, dye_weights=None):
    '''
    Aligns a dye trajectory onto a pdb file, removing any conformations 
    that overlap with protein atoms.
    
    Attributes
    --------------
    pdb : zip(md.Trajectory, state#) 
        PDB of protein conformation, number to label your state for output
    dye: md.Trajectory, 
        Trajectory of dye conformations
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
    pdb, centern = pdb
    
    #Align the dye to the supplied resseq and update xyzs
    dye.xyz=dyefs.align_dye_to_res(pdb, dye.xyz, resseq)

    #Remove conformations that overlap with protein
    dye_indicies = remove_touches_protein_dye_traj(pdb, dye, resseq)
    
    #Optionally, weight the dye indicies
    if len(dye_weights)>1:
        dye_weights=dye_weights[dye_indicies]
        dye_probs = dye_weights / sum(dye_weights)
        
    #Optionally, save the aligned dye structures
    if save_aligned_dyes:
        os.makedirs(f'{outpath}/dye-alignments',exist_ok=True)
        dye[dye_indicies].save_dcd(
            f'{outpath}/dye-alignments/{"".join(dyename.split(" "))}-center-{centern}-residue{resseq}.dcd')
    
    #Pull out the dye emission center and dipole moment for each frame
    dye_r_mu=assemble_dye_r_mu(dye[dye_indicies], dyename, dyelibrary)
    
    return(dye_r_mu)

def map_dye_on_protein(trj, dyename, resseq, outpath='.', save_aligned_dyes=False, weight_dyes=False, n_procs=1):
    '''
    Aligns a dye trajectory onto a pdb file, removing any conformations 
    that overlap with protein atoms.
    
    Attributes
    --------------
    trj : md.Trajectory, 
        Trajectory of protein conformations to map dyes on
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
    
    #Load the dye trajectory
    dye = load_dye(dyename, dyelibrary, dyes_dir)
    
    #Load dye weights (if using)
    if weight_dyes:
        dye_weights=np.loadtxt(
            f'{dye_dir}/weights/{dyelibrary[dyename]["filename"].split("_cutoff")[0]}_cutoff10_weights.txt')
    else:
        dye_weights=[]
    
    #Map the dyes
    func = partial(
        _map_dye_on_protein, dye=dye, resseq=resseq, dyename=dyename, outpath=outpath, 
        save_aligned_dyes=save_aligned_dyes, dye_weights=dye_weights)
    
    pool = Pool(processes=n_procs)
    outputs = pool.map(func, zip(trj, np.arange(len(trj))))
    pool.terminate()
    
    dye_coords = enspara.ra.RaggedArray(outputs)
    
    return(dye_coords)

def find_dyeless_states(dye_coords):
    '''
    Iterates through a ragged array finding empty lists
    
    Attributes
    -----------
    dye_coords, ra.array
        ragged array of all mapped dye positions for
        the cluster centers
    
    Returns
    -----------
    bad_states, np.array, int
        indicies of states with no dye positions mapped
    '''
    
    bad_states=[]
    for i in range(len(dye_coords)):
        if len(dye_coords[i])==0:
            bad_states.append(i)
    
    return(np.array(bad_states))

def remove_bad_states(bad_states, eq_probs, t_probs):
    '''
    Removes bad states from the MSM without re-normalizing.
    
    Crude, probably better to check if states are
    now disconnected and also re-normalize.
    
    Attributes
    -----------
    bad_states, np.array
        indicies of bad states in the MSM
    eq_probs, np.array
        equilibrium probabilities for a MSM
    t_probs, np.array
        transition probabilities for a MSM
    
    Returns
    -----------
    eq_probs, np.array
        eq_probs with bad state indicies 0'd
    t_probs, np.array
        t_probs, with bad states/state transitions 0'd
    '''
    
    eprbs = np.copy(eq_probs)
    tprbs = np.copy(t_probs)

    #Check to see if no bad states
    if len(bad_states)==0:
        return(eprbs,tprbs)
    
    else:
        eprbs[bad_states]=0
        tprbs[:,bad_states]=0
        tprbs[bad_states,:]=0
        return(eprbs, tprbs)

def remove_dyeless_msm_states(dye_coords1, dye_coords2, dyename1, dyename2, eq_probs, t_probs):
    '''
    Removes bad states from the MSM without re-normalizing.
    
    Crude function, probably better to check if states are
    now disconnected and also re-normalize.
    
    Attributes
    -----------
    dye_coords1, ra.RaggedArray
        Mapped dye coordinates/vectors for each state in MSM
    dye_coords2, ra.RaggedArray
        Mapped dye coordinates/vectors for each state in MSM
    dyename1, string
        Name of first dye (only used for notekeeping)
    dyename2, string
        Name of second dye (only used for notekeeping)
    eq_probs, np.array
        equilibrium probabilities for a MSM
    t_probs, np.array
        transition probabilities for a MSM
    
    Returns
    -----------
    eq_probs, np.array
        eq_probs with bad state indicies 0'd
    t_probs, np.array
        t_probs, with bad states/state transitions 0'd
    '''
    
    print(f'Removing states with no available dye-conformations for dye: {dyename1}')
    
    #Get bad_states
    bad_states1 = find_dyeless_states(dye_coords1)

    #Remove any states without dyes mapped (steric clashes)
    eprbs, tprbs = remove_bad_states(bad_states1,eq_probs,t_probs)

    print(f'{len(bad_states1)} states had no availabile dye configuration for dye {dyename1}.')
    print(f'Lost eq_probs of: {np.round(100*(1-eprbs.sum()),3)}% \n')
    

    #Repeat for second dye pair.
    print(f'Removing states with no available dye-conformations for dye: {dyename2}')
    
    Remaining_eq_probs=eprbs.sum()
    #Get bad_states
    bad_states2 = find_dyeless_states(dye_coords2)
    
    #Remove states without dye mappings
    eprbs, tprbs=remove_bad_states(bad_states2,eprbs,tprbs)
    print(f'{len(bad_states2)} states had no availabile dye configuration for dye {dyename2}.')
    print(f'Lost additional eq_probs of: {np.round(100*(Remaining_eq_probs-eprbs.sum()),3)}%')
    print(f'After pruning for both dyes, remaining eq probs is: {np.round(100*(eprbs.sum()),3)} %.')

    #Also return modified dye_coordinates
    bad_states = np.unique(np.concatenate([bad_states1,bad_states2]))
    print(f'Total states removed: {len(bad_states)}/{len(eq_probs)}.')

    for i in bad_states:
        #Fill in all zeros so we keep the array intact but have an obvious mark.
        dye_coords1[i]=[np.zeros(9)]
        dye_coords2[i]=[np.zeros(9)]

    return(eprbs, tprbs, dye_coords1, dye_coords2)