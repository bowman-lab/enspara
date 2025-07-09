import mdtraj as md
import yaml
import enspara
import os
import pandas as pd
import numpy as np
import scipy
from numpy.linalg import norm
from enspara.msm.synthetic_data import synthetic_trajectory
from enspara.geometry import dyes_from_expt_dist as dyefs
from functools import partial
from multiprocessing import Pool
from multiprocessing import get_context

def load_dye(dyename, dyelibrary, dyes_dir):
    """
    Helper function for loading dyes from the enspara dye library.
    """
    
    dye_file=dyelibrary[dyename]["filename"].split("_cutoff")[0]

    #Load the dye and dye weights
    dye=md.load(dyes_dir+f'/trajs/{dye_file}_cutoff10.dcd',top=dyes_dir+f'/structures/{dye_file}.pdb')
    return(dye)

def load_library():
    dyes_dir=os.path.dirname(enspara.__file__)+'/data/dyes'

    with open(f'{dyes_dir}/libraries.yml','r') as yaml_file:
        dyelibrary = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return dyelibrary

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
    Td : float,
        Lifetime of the donor dye in the absence of acceptor (ns)
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
                                   names=['Type', 'Chromophore', 'Ext_coeff', 'QD', 'Td'])
    
    #Pull Quantum yield of the donor absent acceptor
    QD = chromophore_data['QD'].loc[(chromophore_data['Chromophore'] == donor_number) &
                            (chromophore_data['Type'] == donor_fluor)].values.astype(float)

    #Pull donor lifetime in the absence of acceptor
    Td = chromophore_data['Td'].loc[(chromophore_data['Chromophore'] == donor_number) &
                            (chromophore_data['Type'] == donor_fluor)].values.astype(float)
    
    #Pull max extinction coefficient for the acceptor
    ext_coeff_max = chromophore_data['Ext_coeff'].loc[(chromophore_data['Chromophore'] == acceptor_number) &
                            (chromophore_data['Type'] == acceptor_fluor)].values.astype(float)
    
   # Extinction coefficient spectrum of the acceptor
    ext_coeff_acceptor = (ext_coeff_max * acceptor_spectrum['Excitation']).fillna(0)

    # Integral of the donor emission spectrum
    donor_spectra_integral = np.trapezoid(donor_spectrum['Emission'], x=donor_spectrum['Wavelength'])
    
    # Overlap integral between donor-acceptor (normalized by the donor emission spectrum)
    J = np.trapezoid(donor_spectrum['Emission'] * ext_coeff_acceptor * donor_spectrum['Wavelength'] ** 4,
             x=donor_spectrum['Wavelength']) / donor_spectra_integral
    
    return(J, QD, Td)

def remove_touches_protein_dye_traj(pdb, dye, resseq, probe_radius=0.04, atom_tol=6):
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
        radius of a probe to fit between other atom shells to see 
        whether residues are overlapping in nm.
    atom_tol: int,
        Number of overlapping atoms tolerated before a dye is "too clashed"
        to include. 
    
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
        atoms_not_touching_protein>=len(dye.xyz[0])-atom_tol)[0]
    
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
    #xyz is in nm
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

def sample_dye_coords(donor_coords, acceptor_coords, states):
    """
    Picks random dye coordinates for a trj, returns the corresponding k2 and r

    Attributes
    --------------
    Donor_coords, nd.array (9,)
        numpy array specifying the xyz of the dye emission/excitation center,
        the origin of the dipole moment, and the dipole vector
    Acceptor_coords, nd.array (9,)
        numpy array specifying the xyz of the dye emission/excitation center,
        the origin of the dipole moment, and the dipole vector
    states, nd.array, int, (num_states)
        numpy array specifying states to sample dye positions of.

    Returns
    --------------
    k2s : nd.array, float (num_states)
        kappa squared value for the sampled donor/acceptor positions
    rs : nd.array, float (num_states)
        distances between the dye-emission centers for the sampled positions.
    """

    rs, k2s = [], []
    for state in states:
        D_coords=donor_coords[state][np.random.choice(len(donor_coords[state]))]
        A_coords=acceptor_coords[state][np.random.choice(len(acceptor_coords[state]))]
        k2_r=calc_k2_r(D_coords,A_coords)
        k2s.append(k2_r[0])
        rs.append(k2_r[1])
    return np.array(k2s), np.array(rs)


def calc_k2_r(Donor_coords, Acceptor_coords):
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
    r : float,
        distance between the donor and acceptor emission centers (nm)
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
    return(k2, r)

def align_full_dye_to_res(pdb, dye, resseq, dyename, dyelibrary):
    """
    Aligns a dye trajectory to a specific residue using backbone and CB.

    Attributes
    --------------
    pdb : md.Trajectory 
        MDtraj trajectory of protein conformation to align to
    dye: md.Trajectory, 
        MDtraj trajectory of dye conformations
    resseq: int
        residue to label (using PDB ID)
    dyename: string
        name of the dye being added
    dyelibrary: dictionary of dyes
        Must have entry for CB of the dye if you are labeling
        a residue other than GLY or PRO.

    Returns
    ---------------
    dye.xyz : nd.array of aligned atom positions for trajectory
    """

    #This is a lot of work, but some residues are otherwise out of order..
    #Get the residue name
    resname = pdb.top.atom(pdb.top.select(f'resSeq {resseq}')[0]).residue.name

    dye_ca = dye.top.select('name CA')
    dye_n = dye.top.select('name N')
    dye_c = dye.top.select('name C')
    dye_o = dye.top.select('name O')

    prot_ca = pdb.top.select(f'resSeq {resseq} and name CA')
    prot_n = pdb.top.select(f'resSeq {resseq} and name N')
    prot_c = pdb.top.select(f'resSeq {resseq} and name C')
    prot_o = pdb.top.select(f'resSeq {resseq} and name O')

    #If not gly or pro, align to backbone + CB
    if resname != 'GLY' and resname != "PRO":
        dye_cb = dye.top.select(dyelibrary[dyename]['CB'][0])
        dye_sele = np.concatenate((dye_n, dye_ca, dye_cb, dye_c, dye_o))

        prot_cb = pdb.top.select(f'resSeq {resseq} and name CB')
        prot_sele = np.concatenate((prot_n, prot_ca, prot_cb, prot_c, prot_o))

    #If Gly / Pro just do backbone alignment.
    else:
        dye_sele = np.concatenate((dye_n, dye_ca, dye_c, dye_o))
        prot_sele = np.concatenate((prot_n, prot_ca, prot_c, prot_o))
    
    dye = dye.superpose(pdb, atom_indices = dye_sele, ref_atom_indices = prot_sele)
    return(dye.xyz)

def _map_dye_on_protein(pdb, dye, resseq, dyename, dyelibrary,
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
    weights: bool, default=None
        Weight conformation probability by conformation probability in dye traj?
    
    Returns
    ---------------
    
    '''
    pdb, centern = pdb
    
    #Align the dye to the supplied resseq and update xyzs
    dye.xyz=align_full_dye_to_res(pdb, dye, resseq, dyename, dyelibrary)

    #Remove conformations that overlap with protein
    dye_indicies = remove_touches_protein_dye_traj(pdb, dye, resseq)
    
    #Optionally, weight the dye indicies
    if len(dye_weights)>1:
        dye_weights=dye_weights[dye_indicies]
        dye_probs = dye_weights / sum(dye_weights)
        
    #Optionally, save the aligned dye structures
    if save_aligned_dyes:
        if len(dye_indicies)>0:
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
    weights: bool, default=False
        Weight conformation probability by conformation probability in dye traj?
        Not yet implemented
    
    Returns
    ---------------
    
    '''
    
    dyelibrary = load_library()
    
    #Load the dye trajectory
    dye = load_dye(dyename, dyelibrary, dyes_dir)
    
    #Load dye weights (if using)
    if weight_dyes:
        raise Exception("Dye-weighting not yet implemented")
        # dye_weights=np.loadtxt(
        #     f'{dye_dir}/weights/{dyelibrary[dyename]["filename"].split("_cutoff")[0]}_cutoff10_weights.txt')
    else:
        dye_weights=[]
    st = False
    #Map the dyes
    if st == True:
        for i in zip(trj, np.arange(len(trj))):
            _map_dye_on_protein(i, dye=dye, resseq=resseq, dyename=dyename, dyelibrary=dyelibrary, outpath=outpath,
                    save_aligned_dyes=save_aligned_dyes, dye_weights=dye_weights)
    else:
        func = partial(
            _map_dye_on_protein, dye=dye, resseq=resseq, dyename=dyename, dyelibrary=dyelibrary, outpath=outpath, 
            save_aligned_dyes=save_aligned_dyes, dye_weights=dye_weights)
        with get_context("spawn").Pool(processes=n_procs) as pool:
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

def remove_bad_states(bad_states, t_counts):
    '''
    Removes bad states from the MSM with row re-normalizing.
    
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
    
    t_counts = np.copy(t_counts)

    #Check to see if no bad states
    if len(bad_states)==0:
        return(t_counts)
    
    else:
        t_counts[:,bad_states] = 0
        t_counts[bad_states,:] = 0
        return(t_counts)

def remove_dyeless_msm_states(dye_coords1, dye_coords2, dyename1, dyename2, eq_probs, t_counts):
    '''
    Removes bad states from the MSM without re-normalizing.
    
    Crude, probably better to check if states are now disconnected.
    
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
    t_counts, np.array
        transition counts for your MSM
    
    Returns
    -----------
    eq_probs, np.array
        eq_probs with bad state indicies 0'd
    t_probs, np.array
        t_probs, with bad states/state transitions 0'd
    '''
    
    #Get bad_states
    bad_states1 = find_dyeless_states(dye_coords1)
    print(f'{len(bad_states1)} states had no availabile dye configuration for dye {dyename1}.')

    bad_states2 = find_dyeless_states(dye_coords2)
    print(f'{len(bad_states2)} states had no availabile dye configuration for dye {dyename2}.')

    #Remove any states without dyes mapped (steric clashes)
    bad_states = np.unique(np.concatenate((bad_states1,bad_states2)))

    #Remove states without dye_mappings
    trimmed_t_counts = remove_bad_states(bad_states,t_counts)

    #Rebuild the MSM
    print('Rebuilding MSM using row-normalization')

    counts, tprobs, eqs = enspara.msm.builders.normalize(trimmed_t_counts,calculate_eq_probs=True)

    print(f'Total states removed: {len(bad_states)}/{len(t_counts)}.')
    print(f'During pruning for both dyes, lost total eq probs from original model of:')
    print(f'{np.round(100*(eq_probs[bad_states].sum()),3)} %. \n')
    if len(bad_states)/len(t_counts) > 0.2:
        print('WARNING! Labeling resulted in lots of states lost from your MSM.')

    if eq_probs[bad_states].sum() > 0.2:
        print('WARNING! Labeling at this position resulted in major probability loss.')

    #Also return modified dye_coordinates
    for i in bad_states:
        #Fill in all zeros so we keep the array intact but have an obvious mark.
        dye_coords1[i]=[np.zeros(9)]
        dye_coords2[i]=[np.zeros(9)]

    return(eqs, tprobs, dye_coords1, dye_coords2)

def _simulate_burst_k2(MSM_frames, T, populations, dye_coords1, dye_coords2, J, QD, n=1.333):
    """
    Helper function for sampling FRET distributions. Proceeds as follows:
    1) Generate a trajectory of n_frames determined by the burst length
    2) Pick random dye positions for the states that correspond to photon emissions
    3) Calculate the R0 for each instantaneous dye position given the k2 from the dye positions
    4) Calculate the probability of photon transfer
    5) Average acceptor fluorescence to obtain total FRET efficiency for the burst.
    """
    #Introduce a new random seed in each location otherwise pool with end up with the same seeds.
    rng = np.random.default_rng()

    # determine number of frames to sample MSM
    n_frames = np.amax(MSM_frames) + 1

    # sample transition matrix for trajectory
    initial_state = rng.choice(np.arange(T.shape[0]), p=populations)
    trj = synthetic_trajectory(T, initial_state, n_frames)

    #Pull dye orientations for the synthetic trajectory
    k2s, rs = sample_dye_coords(dye_coords1,dye_coords2,trj[MSM_frames])

    #Calculate the corresponding R0
    R0s = calc_R0(k2s, QD, J, n=n)

    #Convert to FRET efficiencies
    FRET_probs = dyefs.FRET_efficiency(rs, R0s)

    # flip coin for donor or acceptor emisions
    acceptor_emissions = rng.random(FRET_probs.shape[0]) <= FRET_probs

    #Average for final observed FRET
    FRET_val = np.mean(acceptor_emissions)
    
    return FRET_val, trj, k2s, rs

def simulate_burst_k2(MSM_frames, T, populations, dye_coords1, dye_coords2, 
                      dyename1, dyename2, n=1.333, n_procs=1):
    
    #Calculate the dye-properties for the provided dyes.
    J, QD, Td = get_dye_overlap(dyename1, dyename2)
    
    # fill in function values
    sample_func = partial(
        _simulate_burst_k2, T = T, populations = populations, 
        dye_coords1 = dye_coords1, dye_coords2 = dye_coords2, 
        J = J, QD = QD, n=n)
    
    # multiprocess, split into chunks to reduce communication overhead
    pool = Pool(processes=n_procs)
    burst_info = pool.map(sample_func, MSM_frames, 
        chunksize = int(np.ceil(len(MSM_frames)/n_procs)))

    pool.terminate()
    
    #Numpy the output
    burst_info = np.array(burst_info, dtype=object)

    #Separate things out
    FEs = burst_info[:,0]
    trajs = burst_info[:,1]
    k2s = burst_info[:,2]
    rs = burst_info[:,3]
    return(FEs, trajs, k2s, rs)

if __name__ == '__main__':
    pass
