import numpy as np
from enspara.geometry import explicit_r0_calc as r0c
from enspara.geometry import dyes_from_expt_dist as dyes_exp_dist
from enspara.msm import builders, synthetic_data
from scipy.optimize import curve_fit
from enspara.msm.transition_matrices import trim_disconnected
from enspara.ra import ra

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

def explicit_static_dyes(d_name, a_name, d_eqs, a_eqs, d_centers, a_centers, 
                            dye_params, dyelibrary, n_samples=1000, rng_seed=None):

    """
    Resolves dye excitations assuming dyes are static with positions determined by
    equilibrium dye positions from the MSM.
    Attributes
    -------------- 
    d_name : string,
        Name of a flurophore in the enspara dye library
    a_name : string,
        Name of a flurophore in the enspara dye library
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
    dyelibrary: dictionary
        enspara dye library
    rng_seed : int, default = None
        seed for numpy.random.default_rng (for testing)

    Returns
    ---------------
    dye_outcomes : [0, string],
        0 is a placeholder for downstream calculations which expect the number of steps for emission
        string indicates how the dye decayed. radiative = donor emission, energy_transfer = FRET
    """

    #Introduce a new random seed in each location otherwise pool with end up with the same seeds.
    rng=np.random.default_rng(rng_seed)
    
    #Extract dye parameters
    J, Qd, Td = dye_params
    
    # Choose a random starting state
    dstates=rng.choice(np.arange(len(d_eqs)), p=d_eqs, size=n_samples)
    astates=rng.choice(np.arange(len(a_eqs)), p=a_eqs, size=n_samples)

    # Convert centers to dye vectors
    d_coords = r0c.assemble_dye_r_mu(d_centers, d_name, dyelibrary)
    a_coords = r0c.assemble_dye_r_mu(a_centers, a_name, dyelibrary)

    dye_outcomes = []

    for dstate, astate in zip(dstates, astates):
        #Calculate distance, k2, and R0 for the random dye states
        k2, r = r0c.calc_k2_r(d_coords[dstate],a_coords[astate])
        R0 = r0c.calc_R0(k2, Qd, J)

        #Calculate a FRET efficiency based off of the distance and R0.
        FE = dyes_exp_dist.FRET_efficiency(r, R0)

        #Randomly choose whether it was a donor or acceptor emission based on transfer probability
        #returns TRUE if random number is less than transfer probability
        #This represents an acceptor emission
        if rng.random() <= FE:
            outcome='energy_transfer'
        else:
            outcome='radiative'

        dye_outcomes.append([0, outcome]) #return a 0 for lifetime since we didn't calculate that..

    return dye_outcomes

def fully_averaged_explict_dyes(d_name, a_name, d_eqs, a_eqs, d_centers, a_centers, 
                            dye_params, dyelibrary, n_samples=1000, rng_seed=None):

    """
    Resolves dye excitations assuming dyes positions are the average of all possible dye positions.

    Attributes
    -------------- 
    d_name : string,
        Name of a flurophore in the enspara dye library
    a_name : string,
        Name of a flurophore in the enspara dye library
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
    dyelibrary: dictionary
        enspara dye library

    Returns
    ---------------
    Lifetimes : int (0), len n_samples
        Placeholder since downstream code depends on lifetimes being present
    transfers : str, len n_samples
        photon identity based on the probability of FRET transfer for the state
    k2s : np.array, len(dstates) * len(astates)
        k2 for each combination of donor/acceptor flurophores
    FEs : np.array, len(dstates) * len(astates)
        FRET Efficiency for each donor/acceptor pair
    eqs : np.array, len(dstates) * len(astates)
        equilibrium probability of each donor x acceptor state.
    """

    #Introduce a new random seed in each location otherwise pool with end up with the same seeds.
    rng=np.random.default_rng(rng_seed)
    
    #Extract dye parameters
    J, Qd, Td = dye_params
    
    # Find the indices of the non-clashing dyes
    dstates = np.where(d_eqs !=0 )[0]
    astates = np.where(a_eqs != 0)[0]

    # Convert centers to dye vectors
    d_coords = r0c.assemble_dye_r_mu(d_centers, d_name, dyelibrary)
    a_coords = r0c.assemble_dye_r_mu(a_centers, a_name, dyelibrary)

    k2s, rs, FEs, eqs = [], [], [], []

    for dstate in dstates:
        for astate in astates:
            #Loop over every acceptor and donor state
            #Calculate each pairwise k2, r, corresponding R0, and resulting FRET probability
            k2, r = r0c.calc_k2_r(d_coords[dstate],a_coords[astate])
            R0 = r0c.calc_R0(k2, Qd, J)

            #Calculate a FRET efficiency based off of the distance and R0.
            FE = dyes_exp_dist.FRET_efficiency(r, R0)

            #Calculate the combined probability of the state 
            eq = d_eqs[dstate]*a_eqs[astate]

            #Save the data
            k2s.append(k2)
            rs.append(r)
            FEs.append(FE)
            eqs.append(eq)

    k2s=np.array(k2s).reshape(-1)
    rs=np.array(rs).reshape(-1)
    FEs = np.array(FEs).reshape(-1)
    eqs = np.array(eqs).reshape(-1)
    avg_FE = np.average(FEs, weights=eqs)

    #Randomly choose whether it was a donor or acceptor emission based on transfer probability
    #returns TRUE if random number is less than transfer probability
    #This represents an acceptor emission

    #choose n_samples random samples. Multiply to convert to 0/1 representation where 0 = no transfer
    transfers = np.multiply(rng.random(n_samples) <= FE, 1, dtype='O')
    transfers[transfers==0] = 'radiative'
    transfers[transfers==1] = 'energy_transfer'

    Lifetimes = [0]*n_samples # save a dummy lifetimes since that's used in downstream code..

    return [Lifetimes, transfers, k2s, FEs, eqs]



def resolve_excitation(d_name, a_name, d_tprobs, a_tprobs, d_eqs, a_eqs, 
                        d_centers, a_centers, dye_params, dye_lagtime, dyelibrary,
                        rng_seed=None):

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
    dyelibrary: dictionary
        enspara dye library
    rng_seed : int, default = None
        seed for numpy.random.default_rng (for testing)

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
    rng=np.random.default_rng(rng_seed)
    
    #Extract dye parameters, calculate constant transfer rates
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

def make_dye_msm(centers, t_counts, pdb, resseq, dyename, dyelibrary, 
    center_n=None, outdir='./', save_dye_xtc = False):
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
    center_n, int
        Protein center_number being labeled (for bookkeeping)
    outdir, path
        where to save output (optional)
    save_dye_xtc, bool default = False
        Save an XTC of the dye positions?
    
    Returns
    -----------
    tprobs, np.array (n_centers, n_centers)
        Transition probabilities for the dye MSM
    eqs, np.array (n_centers,)
        Equilibrium probabilities for the dye MSM
    dye_indicies, np.array len(dyes without steric clash)
        Indicies of centers that don't have steric clashes.
    """
    
    # Align dye to PDB structure
    centers.xyz = r0c.align_full_dye_to_res(pdb, centers, resseq, dyename, dyelibrary)
    
    # Find dye positions with no steric clashes
    dye_indicies = r0c.remove_touches_protein_dye_traj(pdb, centers, resseq)
    
    if len(dye_indicies)==0:
        #No non-clashing label positions
        return np.array([0]),np.array([0]), np.array([])
    
    if save_dye_xtc:
        #Need to think of a clever way to convert dtrj steps to the shortened dye xtc.
        centers[dye_indicies].save_xtc(f'{outdir}/center{center_n}-aligned-to-{resseq}-{"".join(dyename.split(" "))}.xtc')
        #centers.save_xtc(f'{outdir}/center{center_n}-aligned-to-{resseq}-{"".join(dyename.split(" "))}.xtc')
    
    #Reverse the indicies to get the bad ones
    all_indicies = np.arange(len(centers))
    bad_indicies = all_indicies[~np.isin(all_indicies,dye_indicies,assume_unique=True)]
    
    #Purge the t_counts of the bad indicies
    new_tcounts = r0c.remove_bad_states(bad_indicies, t_counts)

    #Rebuild the dye MSM
    counts, tprobs, eqs = builders.normalize(new_tcounts,calculate_eq_probs=True)
    
    return(tprobs, eqs, dye_indicies)

def calc_lifetimes(pdb_center_num, d_centers, d_tcounts, a_centers, a_tcounts, resSeqs, dyenames, 
                   dye_lagtime, n_samples=1000, dye_treatment = 'Monte-carlo', outdir='./', save_dye_trj=False,
                   save_dye_msm=False, save_dye_centers=False, save_k2_r2=False, rng_seed=None):

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
        Warning- this can get expensive if very large. 1000 takes ~ 5 minutes/center to run on my computer.
    dye_dynamics, bool, Default=True
        Account for dye dynamics (True) or just take dye positions according to equilibrium probability (False)?
    outdir, path. Default = './'
        Where to save things to.
    save_dye_trj, bool, default=False
        Save a trajectory of the dye conformations that didn't have steric clashes?
    save_dye_msm, bool, default=False
        Save the rebuilt MSM of the dye conformations that didn't have steric clashes?
    rng_seed, int, default=None
        seed for np.rng, for testing!

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
    
    #Model dye onto residue of interest and remake MSM.
    d_tprobs, d_mod_eqs, d_indxs = make_dye_msm(d_centers,d_tcounts, pdb[0], resSeqs[0], dyenames[0], 
        dyelibrary, center_n = center_n, outdir=outdir,save_dye_xtc=save_dye_centers)

    a_tprobs, a_mod_eqs, a_indxs = make_dye_msm(a_centers,a_tcounts, pdb[0], resSeqs[1], dyenames[1], 
        dyelibrary, center_n = center_n, outdir=outdir,save_dye_xtc=save_dye_centers)
    
    #Check if no feasible labeling positions
    if np.sum(a_mod_eqs) == 0 or np.sum(d_mod_eqs) == 0:
        #return an empty list, could not label one of the positions.
        return [],[]
    
    if save_dye_msm:
        np.save(f'{outdir}/center{center_n}-{"".join(dyenames[0].split(" "))}-{resSeqs[0]}-eqs.npy',d_mod_eqs)
        np.save(f'{outdir}/center{center_n}-{"".join(dyenames[1].split(" "))}-{resSeqs[1]}-eqs.npy',a_mod_eqs)
        np.save(f'{outdir}/center{center_n}-{"".join(dyenames[0].split(" "))}-{resSeqs[0]}-tps.npy',d_tprobs)
        np.save(f'{outdir}/center{center_n}-{"".join(dyenames[1].split(" "))}-{resSeqs[1]}-tps.npy',a_tprobs)

    if dye_treatment == 'Monte-carlo':
        events = np.array([resolve_excitation(dyenames[0], dyenames[1], d_tprobs, a_tprobs, d_mod_eqs, a_mod_eqs, 
                            d_centers, a_centers, dye_params, dye_lagtime, dyelibrary, rng_seed) for i in range(n_samples)], dtype='O')
        
        if save_dye_trj:
            #Dyes are reindexed, events are original indexing. Search to find the 
            #corresponding value in the reindexed array.
            if len(d_indxs) > 0:
                dtrj = np.array([np.searchsorted(d_indxs, event) for event in events[:,2]])
                np.save(f'{outdir}/center{center_n}-{dyenames[0]}-{resSeqs[0]}-dtrj.npy',dtrj)
            if len(a_indxs) > 0:
                atrj = np.array([np.searchsorted(a_indxs, event) for event in events[:,3]])
                np.save(f'{outdir}/center{center_n}-{dyenames[1]}-{resSeqs[1]}-atrj.npy',atrj)
        lifetimes = events[:,0]
        outcomes = events[:,1]

    elif dye_treatment == 'static':
        events = np.array(explicit_static_dyes(dyenames[0], dyenames[1], d_mod_eqs, a_mod_eqs, 
                            d_centers, a_centers, dye_params, dyelibrary, n_samples, rng_seed), dtype='O')

        lifetimes = events[:,0]
        outcomes = events[:,1]

    elif dye_treatment == 'isotropic':
        lifetimes, outcomes, k2s, FEs, eqs = np.array(fully_averaged_explict_dyes(dyenames[0], dyenames[1], d_mod_eqs, a_mod_eqs, 
                            d_centers, a_centers, dye_params, dyelibrary, n_samples, rng_seed), dtype='O')

        if save_k2_r2:
            np.save(f'{outdir}/{resSeqs[0]}-{resSeqs[1]}-per_state_k2s.npy', k2s)
            np.save(f'{outdir}/{resSeqs[0]}-{resSeqs[1]}-per_state_FEs.npy', FEs)
            np.save(f'{outdir}/{resSeqs[0]}-{resSeqs[1]}-per_state_eqs.npy', eqs)


    lifetimes = np.array(lifetimes, dtype=float)*dye_lagtime #ns
    
    return lifetimes, outcomes

def _sample_lifetimes_guarenteed_photon(states, lifetimes, outcomes, rng_seed=None):
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
    rng_seed, int, default=None
        seed for np.rng, for testing.

    Returns
    -----------
    photons, np.array (n_states)
        Observations of acceptor photon (1) or donor photon (0)
    lifetime, np.array (n_states)
        Time since excitation that photon was observed
    """

    rng=np.random.default_rng(rng_seed)

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
            print(f'For reference, state was {state}, and event number {event_n}', flush=True)
            exit()
        lifetime.append(lifetimes[state][event_n])

    photons = np.array(photons)
    lifetime = np.array(lifetime)
    return(photons, lifetime)

def sample_lifetimes_guarenteed_photon(frames, t_probs, eqs, lifetimes, outcomes, rng_seed=None):

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
        Equilibrium probabilities of the protein MSM
    lifetimes, ragged np.array (n_centers, n_samples (or 0))
        Lifetimes of photon excitement for each protein MSM center
    outcomes, ragged np.array (n_centers, n_samples (or 0))
        Outcome of dye excitation (matched with lifetimes, above).
    rng_seed, int, default=None
        random seed for np.rng (for testing purposes)

    Returns
    -----------
    photons, np.array (n_states)
        Observations of acceptor photon (1) or donor photon (0)
    lifetime, np.array (n_states)
        Time since excitation that photon was observed
    """

    #Introduce a new random seed in each location otherwise pool with end up with the same seeds.
    rng=np.random.default_rng(rng_seed)

    # determine number of frames to sample MSM
    n_frames = np.amax(frames) + 1

    # sample transition matrix for trajectory
    initial_state = rng.choice(np.arange(t_probs.shape[0]), p=eqs)    

    #Build a synthetic trajectory from the MSM
    trj = synthetic_data.synthetic_trajectory(t_probs, initial_state, n_frames)

    #Pull lifetimes and outcomes for each MSM frame
    photons, lifetimes = _sample_lifetimes_guarenteed_photon(trj[frames],lifetimes,outcomes)

    return photons, lifetimes, trj[frames]

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
        Equilibrium probabilities of protein MSM, for nice bookkeeping.
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

    print(f'\n{len(bad_states)} of {len(prot_tcounts)} protein states had steric clashes for labeling pair: {resSeqs[0]}-{resSeqs[1]}.',
        flush=True)

    if len(bad_states)/len(prot_tcounts) > 0.2:
        print(f'WARNING! Labeling resulted in loss of {np.round(100*len(bad_states)/len(prot_tcounts))}%') 
        print(f'of your MSM states for labeling pair: {resSeqs[0]}-{resSeqs[1]}. \n', flush=True)

    if prot_eqs is not None:
        if len(bad_states) == 0:
            print(f'No equilibrium probability lost for labeling pair: {resSeqs[0]}-{resSeqs[1]}.')
        else:
            print(f'This was {np.round(100*np.sum(prot_eqs[bad_states]),2)}% of the original equilibrium probability')
            print(f'for labeling pair: {resSeqs[0]}-{resSeqs[1]}.')

            if np.sum(prot_eqs[bad_states]) > 0.2:
                print(f'WARNING! Lots of equilibrium probability lost. \n',flush=True)

    print(f'Remaking MSM for labeling pair: {resSeqs[0]}-{resSeqs[1]}.',flush=True)
    # remove bad states from protein MSM
    trimmed_tcounts = r0c.remove_bad_states(bad_states, prot_tcounts)

    #remake protein MSM
    new_tcounts, new_tprobs, new_eqs = builders.normalize(trimmed_tcounts, calculate_eq_probs=True)

    print(f'Saving modified MSM here: {outdir}.', flush=True)
    np.save(f'{outdir}/{resSeqs[0]}-{"".join(dyenames[0].split(" "))}-{resSeqs[1]}-{"".join(dyenames[1].split(" "))}-eqs.npy',new_eqs)
    np.save(f'{outdir}/{resSeqs[0]}-{"".join(dyenames[0].split(" "))}-{resSeqs[1]}-{"".join(dyenames[1].split(" "))}-t_prbs.npy',new_tprobs)
    return new_tprobs, new_eqs

def remake_msms(resSeq, prot_tcounts, dye_dir, dyenames, orig_eqs, outdir):
    lifetime_outcomes_path = f'{dye_dir}/events-{resSeq[0]}-{resSeq[1]}.npy'

    #Load simulated events
    lifetime_outcomes = np.load(lifetime_outcomes_path, allow_pickle=True)

    #Parse the outcomes
    lifets = lifetime_outcomes[:,0]
    outcomes = lifetime_outcomes[:,1]

    #Remake the protein MSM
    new_tprobs, new_eqs = remake_prot_MSM_from_lifetimes(lifets, prot_tcounts, 
                                                resSeq, dyenames, outdir= f'{outdir}/MSMs', prot_eqs = orig_eqs)

def run_mc(resSeq, prot_tcounts, dyenames, MSM_frames, dye_dir, outdir, time_correction, 
    save_photon_trjs=False, save_burst_frames=False):
    import os
    
    lifetime_outcomes_path = f'{dye_dir}/events-{resSeq[0]}-{resSeq[1]}.npy'

    #Load simulated events
    lifetime_outcomes = np.load(lifetime_outcomes_path, allow_pickle=True)

    #Parse the outcomes
    lifets = lifetime_outcomes[:,0]
    outcomes = lifetime_outcomes[:,1]

    new_tprobs = np.load(f'{outdir}/MSMs/{resSeq[0]}-{"".join(dyenames[0].split(" "))}-{resSeq[1]}-{"".join(dyenames[1].split(" "))}-t_prbs.npy')
    new_eqs = np.load(f'{outdir}/MSMs/{resSeq[0]}-{"".join(dyenames[0].split(" "))}-{resSeq[1]}-{"".join(dyenames[1].split(" "))}-eqs.npy')

    print(f'Running MC sampling for {resSeq[0]}-{resSeq[1]} and time factor {time_correction}.', flush=True)
    #Sample the protein MSM to get bursts
    sampling = np.array([
        sample_lifetimes_guarenteed_photon(
        frames, new_tprobs, new_eqs, lifets, outcomes) for frames in MSM_frames], dtype='O')

    if save_burst_frames:
        os.makedirs(f'{outdir}/protein-trajs/', exist_ok=True)
        np.save(f'{outdir}/protein-trajs/{resSeq[0]}-{resSeq[1]}-{time_correction}.npy', sampling[:,2])

    print(f'Extracting FEs and lifetimes for {resSeq[0]}-{resSeq[1]} and time factor {time_correction}.', flush=True)

    FEs, d_lifetimes, a_lifetimes = extract_fret_efficiency_lifetimes(
        sampling)

    print(f'Saving results for {resSeq[0]}-{resSeq[1]} and time factor {time_correction}.', flush=True)

    #Convert from photons to FRET E
    FEs = np.array([np.sum(FE)/len(FE) for FE in sampling[:,0]])
    os.makedirs(f'{outdir}/Lifetimes', exist_ok=True)
    os.makedirs(f'{outdir}/FEs', exist_ok=True)
    if save_photon_trjs:
        photon_ids = ra.RaggedArray([burst for burst in sampling[:,0]])
        ra.save(f'{outdir}/FEs/photon-trace-{resSeq[0]}-{resSeq[1]}-{time_correction}.h5', photon_ids)
    np.save(f'{outdir}/FEs/FE-{resSeq[0]}-{resSeq[1]}-{time_correction}.npy', FEs)
    np.save(f'{outdir}/Lifetimes/d_lifetimes-{resSeq[0]}-{resSeq[1]}-{time_correction}.npy', d_lifetimes)    
    np.save(f'{outdir}/Lifetimes/a_lifetimes-{resSeq[0]}-{resSeq[1]}-{time_correction}.npy', a_lifetimes)

def calc_per_state_FE(events):
    """
    Takes an events array from calc_lifetimes and returns the FRET efficiency per protein state.

    Attributes
    ____________
    events, np.arrray shape (n_states, 2, n_samples)
        Dye lifetimes and outcomes array from calc_lifetimes

    Returns
    ____________
    per_state, np.array shape (n_states)
        FRET efficiency for each state, averaged over the number of samples.
    """
    per_state=[]
    for event in events[:,1]:
        if len(event)==0:
            #If state had no label pairs, return that the FE is nan
            per_state.append(np.nan)
        else:
            acceptors = np.count_nonzero(event=='energy_transfer')
            donors = np.count_nonzero(event=='radiative')
            per_state.append(acceptors/(donors+acceptors))

    return np.array(per_state)

def single_exp_decay(t, Io, tau):
    """
    Function for a single exponential decay.
    Attributes
    -------------- 
    t : np.array
        Time
    Io : float
        Initial maximum
    tau : float
        lifetime
    """
    
    return Io*np.exp(-t/tau)

def fit_single_exp(t,y,p0):
    """
    Fits a single exponential decay curve to data, returns optimum parameters.
    """
    opt_params, parm_cov = curve_fit(single_exp_decay, t, y, p0=p0)
    Io, tau = opt_params
    return Io, tau

def fit_lifetimes_single_exp(lifetimes, donor_name=None, hist_bins = 100, hist_range=(0,25)):
    """
    Fits decay lifetimes to a single exponential decay making reasonable initial guesses
    
    Attributes
    -------------- 
    lifetimes : np.array, shape (n_lifetimes,)
        Lifetimes of dye
    donor name : string, default = None
        Dye in enspara library. Used to get initial guess for lifetime
        Makes more accurate initial guess. If not passed, we make an ok initial guess.
    hist_bins : int, defaults = 100
        How many bins to histogram lifetimes into?
    hist_range: tuple of ints, default = (0,25)
        What range should lifetimes be histogrammed over?
    
    Returns
    -------------
    t : np.array
        Lifetime histogram bin center values
    counts : np.array
        Counts associated with histogram bins
    fit_I : float
        Initial amplitude of decay
    fit_tau : float
        Lifetime of the decay
    """
    
    #Histogram the lifetimes
    counts, edges = np.histogram(lifetimes, range=hist_range, bins=hist_bins)

    bin_w = edges[1]-edges[0]
    t = edges[:-1]+bin_w/2

    #Guess initial parameters
    #Only going to use this to pull donor lifetime, so can pass donor name twice
    if donor_name==None:
        Td = 4 #reasonable lifetime guess given most single molecule dyes.
    else:
        J, QD, Td = r0c.get_dye_overlap(donor_name, donor_name)
    
    Io = np.amax(counts)

    fit_I, fit_tau = fit_single_exp(t, counts, p0 = np.array([Io, Td[0]]))
    
    return(t, counts, fit_I, fit_tau)

def double_exp_decay(t, Io1, Io2, tau1, tau2):
    """
    Function for a single exponential decay.
    Attributes
    -------------- 
    t : np.array
        Time
    Io1 : float
        Initial maximum guess for first decay curve
    Io2 : float
        Initial maximum guess for second decay curve
    tau1 : float
        Initial lifetime guess for first lifetime
    tau2 : float
        Initial lifetime guess for second lifetime
    """
    return Io1*np.exp(-t/tau1) + Io2*np.exp(-t/tau2)

def fit_double_exp(t,y,p0):
    """
    Fits a double exponential decay curve to data, returns optimum parameters.
    """ 
    opt_params, parm_cov = curve_fit(double_exp_decay, t, y, p0=p0)
    Io1, Io2, tau1, tau2 = opt_params
    return Io1, Io2, tau1, tau2

def fit_lifetimes_double_exp(lifetimes, donor_name=None, hist_bins = 100, hist_range=(0,25)):
    """
    Fits decay lifetimes to a double exponential decay making reasonable initial guesses
    
    Attributes
    -------------- 
    lifetimes : np.array, shape (n_lifetimes,)
        Lifetimes of dye
    donor name : string, default = None
        Dye in enspara library. Used to get initial guess for lifetime
        Makes more accurate initial guess. If not passed, we make an ok initial guess.
    hist_bins : int, defaults = 100
        How many bins to histogram lifetimes into?
    hist_range: tuple of ints, default = (0,25)
        What range should lifetimes be histogrammed over?
        
    Returns
    -------------
    t : np.array
        Lifetime histogram bin center values
    counts : np.array
        Counts associated with histogram bins
    fit_I1 : float
        Initial amplitude of first decay curve7
    fit_I2 : float
        Initial amplitude of second decay curve
    fit_tau1 : float
        Lifetime of the first decay
    fit_tau2 : float
        Lifetime of the second decay
    """
    
    #Histogram the lifetimes
    counts, edges = np.histogram(lifetimes, range=hist_range, bins=hist_bins)

    bin_w = edges[1]-edges[0]
    t = edges[:-1]+bin_w/2

    #Guess initial parameters
    #Only going to use this to pull donor lifetime, so can pass donor name twice
    if donor_name==None:
        Td = 4 #reasonable lifetime guess given most single molecule dyes.
    else:
        J, QD, Td = r0c.get_dye_overlap(donor_name, donor_name)
        
    Io = np.amax(counts)

    fit_I1, fit_I2, fit_tau1, fit_tau2 = fit_double_exp(t, counts, p0 = np.array([Io/2, Io/2, Td[0], Td[0]]))
    
    return(t, counts, fit_I1, fit_I2, fit_tau1, fit_tau2)

def extract_fret_efficiency_lifetimes(lifetime_samples):
    """
    Extracts FRET efficiency and donor/acceptor lifetimes from 
    sample_lifetimes_guarenteed_photon arrays.
    
    Attributes
    -------------- 
    lifetime_samples : np.array, shape (n_bursts, 2, variable)
        Ragged array of photons and lifetimes for each burst.
        Should be able to directly pass the output of repeated calls
        to sample_lifetimes_guarenteed_photon to this!
        
    Returns
    -------------
    FEs : np.array, shape (n_bursts)
        Average lifetime for each burst
    d_lifetimes : np.array (n_bursts, variable)
        Lifetimes associated with each donor photon in a burst.
    a_lifetimes : np.array (n_bursts, variable)
        Lifetimes associated with each donor photon in a burst.
    """
    
    FEs = np.array([np.sum(burst)/len(burst) for burst in lifetime_samples[:,0]])
    
    d_lifetimes, a_lifetimes=[],[]
    for burst in lifetime_samples:
        d_lifetimes.append(burst[1][np.where(burst[0]==0)[0]])
        a_lifetimes.append(burst[1][np.where(burst[0]==1)[0]])

    d_lifetimes=np.array(d_lifetimes, dtype=object)
    a_lifetimes=np.array(a_lifetimes, dtype=object)
    return FEs, d_lifetimes, a_lifetimes

def fit_lifetimes_single_exp_high_throughput(lifetimes, donor_name=None, hist_bins = 100, hist_range=(0,25)):
    """
    Fits decay lifetimes to a single exponential decay making reasonable initial guesses
    Some run-time handling in case of bad fitting (returns HIGH half life.)
    
    Attributes
    -------------- 
    lifetimes : np.array, shape (n_lifetimes,)
        Lifetimes of dye
    donor name : string, default = None
        Dye in enspara library. Used to get initial guess for lifetime
        Makes more accurate initial guess. If not passed, we make an ok initial guess.
    hist_bins : int, defaults = 100
        How many bins to histogram lifetimes into?
    hist_range: tuple of ints, default = (0,25)
        What range should lifetimes be histogrammed over?
    
    Returns
    -------------
    t : np.array
        Lifetime histogram bin center values
    counts : np.array
        Counts associated with histogram bins
    fit_I : float
        Initial amplitude of decay
    fit_tau : float
        Lifetime of the decay
    """
    
    #Histogram the lifetimes
    counts, edges = np.histogram(lifetimes, range=hist_range, bins=hist_bins)

    bin_w = edges[1]-edges[0]
    t = edges[:-1]+bin_w/2

    #Guess initial parameters
    #Only going to use this to pull donor lifetime, so can pass donor name twice
    if donor_name==None:
        Td = 4 #reasonable lifetime guess given most single molecule dyes.
    else:
        J, QD, Td = r0c.get_dye_overlap(donor_name, donor_name)
    
    Io = np.amax(counts)

    try:
        fit_I, fit_tau = lifetime_fns.fit_single_exp(t, counts, p0 = np.array([Io, Td[0]]))
    except RuntimeError:
        return(t, counts, 0, 100)
    
    return(t, counts, fit_I, fit_tau)
