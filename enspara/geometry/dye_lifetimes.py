import numpy as np
from enspara.geometry import explicit_r0_calc as r0c
from enspara.msm import builders
from enspara.msm import synthetic_data

def FRET_rate(r, R0, Td):
    """
    Calculate the rate of FRET energy transfer as a function of flurophore parameters

    Attributes
    --------------    
    r : float, 
        Distance between donor and acceptor flurophore, nm
    R0 : float,
        Forster radius, nm
    Td : float,
        fluorescence lifetime of donor in absence of acceptor, ns
    
    Returns
    ---------------
    kRET : float,
    rate of FRET transfer 1/ns
    """
    return((1/Td)*((R0/r)**6))

def calc_dye_radiative_rates(Qd, Td):
    """
    Calculates rate of radiative/non_radiative energy transfer given:

    Attributes
    --------------    
    Qd : float,
        fluorescence quantum yield
    Td : float,
        donor lifetime in absence of acceptor

    Returns
    ---------------
    krad : float,
        Rate of radiative energy decay (flurophore emission), 1/ns
    k_non_radiative : float,
        Rate of non radiative decay, 1/ns
    """
    
    krad = Qd/Td
    k_non_rad = (1/Td) - krad
    
    return(krad, k_non_rad)

def calc_energy_transfer_prob(krad, k_non_rad, kRET, dt):
    """
    Calculates probability of energy transfers given a timestep.
    
    Attributes
    -------------- 
    krad : float,
        Rate  of radiative decay, 1/ns
    k_non_rad : float,
        Rate of non-radiative decay, 1/ns
    kRET : float,
        Rate of energy transfer to acceptor, 1/ns
    dt : float,
        Timestep to evaluate probability over, ns

    Returns
    ---------------
    all_probs : np.array (4,)
        Probabilities of occupying any of the decay states or remaining excited.
    """
    
    p_rad = 1 - np.exp(-krad * dt)
    p_nonrad = 1 - np.exp(-k_non_rad * dt)
    p_RET = 1 - np.exp(-kRET * dt)
    p_remain_excited = 1 - p_rad - p_nonrad - p_RET
    all_probs = np.array([p_rad, p_nonrad, p_RET, p_remain_excited])

    # If dyes are very close can get 100% transfer efficiency
    if p_remain_excited < 0:

        p_remain_excited = np.zeros(1)

        all_probs = np.array([p_rad, p_nonrad, p_RET, p_remain_excited])

        all_probs = all_probs / all_probs.sum()
        
    return(all_probs.flatten())


def resolve_excitation(d_name, a_name, d_tprobs, a_tprobs, d_eqs, a_eqs, 
                        d_centers, a_centers, dye_params, dye_lagtime, dyelibrary):

    """
    Runs a Monte Carlo to watch for dye decay and reports back the dye lifetime
    and the decay pathway.

    Attributes
    -------------- 
    d_name : string,
        Name of a flurophore in the enspara dye library
    a_name : string,
        Name of a flurophore in the enspara dye library
    d_tprobs : np.array,
        Transition probabilities from donor dye MSM. Shape (n_states, n_states)
    a_tprobs : np.array,
        Transition probabilities from acceptor dye MSM. Shape (n_states, n_states)
    d_eqs : np.array,
        Equilibrium probabilities from donor dye MSM. shape (n_states)
    a_eqs : np.array,
        Equilibrium probabilities from acceptor dye MSM. shape (n_states)
    d_centers : md.Trajectory,
        MDtraj trajectory of donor dye conformations. shape (n_states)
    a_centers : md.Trajectory,
        MDtraj trajectory of acceptor dye conformations. shape (n_states)
    dye_params : tuple,
        Dye-pair overlap, Dye quantum yield, Donor dye lifetime (absent acceptor)
        Direct output of r0c.get_dye_overlap
    dye_lagtime : float,
        Lagtime for the dye MSMs in ns.

    Returns
    ---------------
    steps : int,
        Number of dye MSM steps it took for decay to occur.
    d_state : string,
        How the dye decayed. radiative = donor emission, energy_transfer = FRET
        non_radiative = no visible decay.
    dtraj : np.array,
        MSM centers visited by the donor dye while dye was excited. Shape, (n_states)
    atraj : np.array,
        MSM centers visited by the acceptor dye while donor was excited. Shape, (n_states)    
    """

    #Introduce a new random seed in each location otherwise pool with end up with the same seeds.
    rng=np.random.default_rng()
    
    J, Qd, Td = dye_params
    
    krad = Qd/Td #Constant
    k_non_rad = (1/Td) - krad #Constant
    
    # Choose a random starting state
    dtrj,atrj =[],[]
    dtrj.append(rng.choice(np.arange(d_tprobs.shape[0]), p=d_eqs))
    atrj.append(rng.choice(np.arange(a_tprobs.shape[0]), p=a_eqs))

    # Convert centers to dye vectors
    d_coords = r0c.assemble_dye_r_mu(d_centers, d_name, dyelibrary)
    a_coords = r0c.assemble_dye_r_mu(a_centers, a_name, dyelibrary)
    
    n_dcenters = len(d_centers)
    n_acenters = len(a_centers)

    # Define potential donor resolution pathways
    dye_outcomes = np.array(['radiative','non_radiative','energy_transfer','excited'])

    #Start up the markov chain

    d_state = 'excited'
    steps = 0

    #Run the markov chain
    while d_state == 'excited':
        #Calculate k2, r, R0, and kRET for new dye position
        k2, r = r0c.calc_k2_r(d_coords[dtrj[steps]],a_coords[atrj[steps]])
        R0 = r0c.calc_R0(k2, Qd, J)
        kRET = FRET_rate(r, R0, Td)

        #Calculate probability of each decay  mechanism
        transfer_probs = calc_energy_transfer_prob(krad, k_non_rad, kRET, dye_lagtime)
        
        #Pick a decay mechanism according to probabilities
        d_state = rng.choice(dye_outcomes, p=transfer_probs)

        #Pick new dye positions based on probability of hopping MSM states
        dtrj.append(
            rng.choice(n_dcenters,p=d_tprobs[dtrj[-1],:]))
        atrj.append(
            rng.choice(n_acenters,p=a_tprobs[atrj[-1],:]))
        
        #Add a new step to our counter
        steps+=1        
            
    return([steps, d_state, np.array(dtrj), np.array(atrj)])

def make_dye_msm(centers, t_counts, pdb, resseq, dyename, dyelibrary, outdir='./', save_dye_xtc = False):
    """
    Labels a protein residue with a given dye and returns a new dye MSM
    Attributes
    -----------
    centers, md.Trajectory
        Trajectory of dye centers
    t_counts, np.array (n_centers, n_centers)
        transition counts from dye MSM
    pdb, md.Trajectory
        structure of protein to label
    resseq, int
        ResSeq number to label of pdb
    dyename, string
        name of dye in enspara dye library
    dyelibrary, dictionary
        enspara dye library
    outdir, path
        where to save output
    save_dye_xtc, bool default = False
        Save an XTC of the dye positions?
    
    Returns
    -----------
    tprobs, np.array (n_centers, n_centers)
        Transition probabilities for the dye MSM
    eqs, np.array (n_centers,)
        Equilibrium probabilities for the dye MSM
    """
    
    # Align dye to PDB structure
    centers.xyz = r0c.align_full_dye_to_res(pdb, centers, resseq, dyename, dyelibrary)
    
    # Remove steric clashes
    dye_indicies = r0c.remove_touches_protein_dye_traj(pdb, centers, resseq)
    
    if len(dye_indicies)==0:
        #No non-clashing label positions
        #Need to think about what to return so we can rebuild the protein MSM correctly.
        return np.array([0]),np.array([0])
    
    if save_dye_xtc:
        centers[dye_indicies].save_xtc(f'{outdir}/{resseq}-{"".join(dyename.split(" "))}.xtc')
    
    #Reverse the indicies to get the bad ones
    all_indicies = np.arange(len(centers))
    bad_indicies = all_indicies[~np.isin(all_indicies,dye_indicies,assume_unique=True)]
    
    #Purge the t_counts of the bad indicies
    new_tcounts = r0c.remove_bad_states(bad_indicies, t_counts)
    
    #Rebuild the dye MSM
    counts, tprobs, eqs = builders.normalize(new_tcounts,calculate_eq_probs=True)
    
    return(tprobs, eqs)


def calc_lifetimes(pdb_center_num, d_centers, d_tcounts, a_centers, a_tcounts, resSeqs, dyenames, 
                   dye_lagtime, n_samples=1000, outdir='./', save_dye_trj=False, save_dye_msm=False):

    """
    Takes a protein pdb structure, dye trajectories/MSM, and labeling positions and calculates expected
    dye-emission event and lifetime for n_samples. Dye is allowed to move during the monte-carlo according
    to the MSM probabilities and transfer probabilities are iteratively updated.

    Attributes
    -----------
    pdb_center_num, zip(md.Trajectory, int)
        PDB to model dyes on. Int is the center number for book keeping.
    d_centers, md.Trajectory, size(n_states)
        MSM centers for the donor dye.
    d_tcounts, np.array (n_centers, n_centers)
        T_counts for donor dye msm.
    a_centers, md.Trajectory, size(n_states)
        MSM centers for the acceptor dye.
    a_tcounts, np.array (n_centers, n_centers)
        T_counts for acceptor dye msm.
    resSeqs, list of ints, len(2)
        resSeqs to label. Donor will go on first residue and acceptor on second.
    dyenames, list of strings, len(2)
        names of dyes to label residues with. Donor is the first dyename and acceptor the second.
        Should be in the enspara dye library.
    dye_lagtime, float
        Lagtime used to build the dye MSMs.
    n_samples, int. Default = 1000
        Number of monte carlo simulations to run.
        Warning- this can get expensive if very large. 1000 takes ~ 5 minutes to run on my computer.
    outdir, path. Default = './'
        Where to save things to.
    save_dye_trj, bool, default=False
        Save a trajectory of the dye conformations that didn't have steric clashes?
    save_dye_msm, bool, default=False
        Save the rebuilt MSM of the dye conformations that didn't have steric clashes?

    Returns
    -----------
    lifetimes, np.array (n_states)
        How long did it take for decay to occur?
    outcomes, np.array (n_states)
        How did the decay occcur? 
        Radiative = donor flurophore emission
        non-radiative = no observed emission
        energy_transfer = acceptor flurophore emission
    """

    dyelibrary = r0c.load_library()
    dye_params = r0c.get_dye_overlap(dyenames[0], dyenames[1])
    
    pdb, center_n = pdb_center_num
    
    #Model dye onto residue of interest and remake MSM. Repeat per labeling position
    d_tprobs, d_mod_eqs = make_dye_msm(d_centers,d_tcounts, pdb[0], resSeqs[0], dyenames[0], dyelibrary)
    a_tprobs, a_mod_eqs = make_dye_msm(a_centers,a_tcounts, pdb[0], resSeqs[1], dyenames[1], dyelibrary)
    
    #Check if no feasible labeling positions
    if np.sum(a_mod_eqs) == 0 or np.sum(d_mod_eqs) == 0:
        #return an empty list, could not label
        return [],[]
    
    if save_dye_msm:
        np.save(f'{outdir}/center{center_n[0][0]}-{"".join(dyenames[0].split(" "))}-eqs.npy',d_mod_eqs)
        np.save(f'{outdir}/center{center_n[0][0]}-{"".join(dyenames[1].split(" "))}-eqs.npy',a_mod_eqs)
        np.save(f'{outdir}/center{center_n[0][0]}-{"".join(dyenames[0].split(" "))}-tps.npy',d_tprobs)
        np.save(f'{outdir}/center{center_n[0][0]}-{"".join(dyenames[1].split(" "))}-tps.npy',a_tprobs)

    
    events = np.array([resolve_excitation(dyenames[0], dyenames[1], d_tprobs, a_tprobs, d_mod_eqs, a_mod_eqs, 
                        d_centers, a_centers, dye_params, dye_lagtime, dyelibrary) for i in range(n_samples)])
    
    if save_dye_trj:
        dtrj = events[:,2]
        atrj = events[:,3]
        np.save(f'{outdir}/center{center_n[0][0]}-{dyenames[0]}-dtrj.npy',dtrj)
        np.save(f'{outdir}/center{center_n[0][0]}-{dyenames[0]}-atrj.npy',atrj)

    lifetimes = events[:,0].astype(float)*dye_lagtime #ns
    outcomes = events[:,1]
    
    return lifetimes, outcomes

def _sample_lifetimes_guarenteed_photon(states, lifetimes, outcomes):
    """
    Samples dye lifetimes/outcomes such as outputs of calc_lifetimes at specific MSM states.
    Returns random, observed lifetime/outcome for that MSM state.
    Guarentees observation of a photon (donor or acceptor) as opposed to non-radiative decay.

    Attributes
    -----------
    states, np.array
        MSM states to pull lifetime/excitation outcomes from
    lifetimes, ragged np.array (n_centers, n_samples (or 0))
        Lifetimes of photon excitement
    outcomes, ragged np.array (n_centers, n_samples (or 0))
        Outcome of dye excitation (matched with lifetimes, above).

    Returns
    -----------
    photons, np.array (n_states)
        Observations of acceptor photon (1) or donor photon (0)
    lifetime, np.array (n_states)
        Time since excitation that photon was observed
    """

    rng=np.random.default_rng()

    photons, lifetime = [],[]
    for state in states:
        event_n = rng.choice(len(lifetimes[state]))

        #If non-radiative, redraw since we're using experimental photon arrival times
        while outcomes[state][event_n]=='non_radiative':
            event_n = rng.choice(len(lifetimes[state]))
        if outcomes[state][event_n]=='energy_transfer':
            photons.append(1)
            #Acceptor event
        elif outcomes[state][event_n]=='radiative':
            photons.append(0)
            #Donor event
        else:
            #Something went wrong.
            print('Something seems wrong with your outcomes array, expected outcomes of:')
            print(f'non_radiative, energy_transfer, or radiative. Got {outcomes[state][event_n]}.')
            print(f'For reference, state was {state}, and event number {event_n}')
            exit()
        lifetime.append(lifetimes[state][event_n])

    photons = np.array(photons)
    lifetime = np.array(lifetime)
    return(photons, lifetime)

def sample_lifetimes_guarenteed_photon(frames, t_probs, eqs, lifetimes, outcomes):

    """
    Samples dye lifetimes and excitation outcomes given protein MSM frames and a protein MSM.
    Guarentees observation of a photon, non-radiative decay events ignored.

    Attributes
    -----------
    frames, np.array shape (n_frames,)
        Steps through MSM when photons are observed
    t_probs, np.array, shape (n_states, n_states)
        Transition probabilities of the protein MSM
    eqs, np.array, shape (n_states,)
        Equilibirum probabilities of the protein MSM
    lifetimes, ragged np.array (n_centers, n_samples (or 0))
        Lifetimes of photon excitement for each protein MSM center
    outcomes, ragged np.array (n_centers, n_samples (or 0))
        Outcome of dye excitation (matched with lifetimes, above).

    Returns
    -----------
    photons, np.array (n_states)
        Observations of acceptor photon (1) or donor photon (0)
    lifetime, np.array (n_states)
        Time since excitation that photon was observed
    """

    #Introduce a new random seed in each location otherwise pool with end up with the same seeds.
    rng=np.random.default_rng()

    # determine number of frames to sample MSM
    n_frames = np.amax(frames) + 1

    # sample transition matrix for trajectory
    initial_state = rng.choice(np.arange(t_probs.shape[0]), p=eqs)    

    #Build a synthetic trajectory from the MSM
    trj = synthetic_data.synthetic_trajectory(t_probs, initial_state, n_frames)

    #Pull lifetimes and outcomes for each MSM frame
    photons, lifetimes = _sample_lifetimes_guarenteed_photon(trj[frames],lifetimes,outcomes)

    return photons, lifetimes

def remake_prot_MSM_from_lifetimes(lifetimes, prot_tcounts, resSeqs, dyenames, outdir='./', prot_eqs=None):
    """
    Rebuilds protein MSM removing states that had steric clashes with dyes.
    Attributes
    -----------
    lifetimes, np.array
        ragged array of dye lifetimes shape (n_prot_states, n_sampled_lifetimes or 0)
    prot_tcounts, np.array (n_centers, n_centers)
        T_counts from protein MSM
    outdir, path
        Where to save data to. Default = ./
    prot_eqs, np.array (n_centers, n_centers)
        Equilibirium probabilities of protein MSM, for nice bookkeeping.
        Default = None

    Returns
    -----------
    new_tprobs, np.array (n_centers, n_centers)
        Transition probabilities for the clash-free protein MSM
    new_eqs, np.array (n_centers,)
        Equilibrium probabilities for the clash-free protein MSM
    """

    # Find which states couldn't be labeled:
    bad_states = r0c.find_dyeless_states(lifetimes)

    print(f'{len(bad_states)} of {len(prot_tcounts)} protein states had steric clashes.')

    if len(bad_states)/len(prot_tcounts) > 0.2:
        print(f'WARNING! Labeling resulted in loss of {np.round(100*len(bad_states)/len(prot_tcounts))}% of your MSM states. \n')

    if prot_eqs is not None:
        print(f'\nThis was {np.round(100*np.sum(prot_eqs[bad_states]),2)}% of the original equilibirum probability.')

        if np.sum(prot_eqs[bad_states]) > 0.2:
            print(f'WARNING! Lots of equilibrium probability lost. \n')

    print(f'Remaking MSM.')
    # remove bad states from protein MSM
    trimmed_tcounts = r0c.remove_bad_states(bad_states, prot_tcounts)

    #remake protein MSM
    #Add in a "you lost this many states / eq probs."
    new_tcounts, new_tprobs, new_eqs = builders.normalize(trimmed_tcounts, calculate_eq_probs=True)

    print(f'Saving modified MSM here: {outdir}.')
    np.save(f'{outdir}/{resSeqs[0]}-{"".join(dyenames[0].split(" "))}-{resSeqs[1]}-{"".join(dyenames[1].split(" "))}-eqs.npy',new_eqs)
    np.save(f'{outdir}/{resSeqs[0]}-{"".join(dyenames[0].split(" "))}-{resSeqs[1]}-{"".join(dyenames[1].split(" "))}-t_prbs.npy',new_tprobs)
    return new_tprobs, new_eqs