import numpy as np
from enspara.msm.synthetic_data import synthetic_trajectory
from enspara.geometry import explicit_r0_calc as r0c


def FRET_rate(r, k2, R0, Td):
    """
    Calculate the rate of FRET energy transfer as a function of flurophore parameters

    Attributes
    --------------    
    r : float, 
        Distance between donor and acceptor flurophore, nm
    k2 : float,
        Dipole moment between donor and acceptor flurophore
    R0 : float,
        Forster radius, nm
    Td : float,
        fluorescence lifetime of donor in absence of acceptor, ns
    
    Returns
    ---------------
    kEET : float,
    rate of FRET transfer 1/ns
    """
    return((1/Td)*(k2)*(R0/r)**6)

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

def calc_energy_transfer_prob(krad, k_non_rad, kEET, dt):
    """
    Calculates probability of energy transfers given a timestep.
    
    Attributes
    -------------- 
    krad : float,
        Rate  of radiative decay, 1/ns
    k_non_rad : float,
        Rate of non-radiative decay, 1/ns
    kEET : float,
        Rate of energy transfer to acceptor, 1/ns
    dt : float,
        Timestep to evaluate probability over, ns

    Returns
    ---------------
    Probabilities of occupying any of the decay states or remaining excited.
    """
    p_rad = 1 - np.exp(-krad * dt)
    p_nonrad = 1 - np.exp(-k_non_rad * dt)
    p_EET = 1 - np.exp(-kEET * dt)
    p_remain_excited = 1 - p_rad - p_nonrad - p_EET
    all_probs = np.concatenate((p_rad, p_nonrad, p_EET, p_remain_excited))
    
    # If dyes are very close can get 100% transfer efficiency
    if p_remain_excited < 0:
        
        p_remain_excited == 0
        
        all_probs = all_probs / all_probs.sum()
        
    return(np.concatenate((p_rad, p_nonrad, p_EET, p_remain_excited)))

def resolve_excitation(d_name, a_name, d_tprobs, a_tprobs, d_eqs, a_eqs, 
                        d_centers, a_centers, dye_params,dye_lagtime):

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
    
    # Avoid repeated calls to synthetic trajectory (expensive) so ask for
    # somewhat long chunks of frames. 0.1 x dye lifetime emperically seems
    # to be the best balance between # of frames to generate and repeated calls.
    n_frames = int( Td * 0.1 / dye_lagtime )
    
    # sample transition matrix for trajectory
    dinitial_state = rng.choice(np.arange(d_tprobs.shape[0]), p=d_eqs)    
    ainitial_state = rng.choice(np.arange(a_tprobs.shape[0]), p=a_eqs)

    dtrj = synthetic_trajectory(d_tprobs, dinitial_state, n_frames)
    atrj = synthetic_trajectory(a_tprobs, ainitial_state, n_frames)

    d_coords = r0c.assemble_dye_r_mu(d_centers, d_name, dyelibrary)
    a_coords = r0c.assemble_dye_r_mu(a_centers, a_name, dyelibrary)


    dye_outcomes = np.array(['radiative','non_radiative','energy_transfer','excited'])

    #Start up the markov chain

    d_state = 'excited'
    steps = 0

    #Run the markov chain
    while d_state == 'excited':
        #Calculate k2, r, R0, and kEET for new dye position

        k2, r = r0c.calc_k2_r(d_coords[steps],a_coords[steps])
        R0 = r0c.calc_R0(k2, Qd, J)
        kEET = FRET_rate(r, k2, R0, Td)

        #Calculate probability of each decay mechanism
        transfer_probs = calc_energy_transfer_prob(krad, k_non_rad, kEET, dye_lagtime)

        d_state = rng.choice(dye_outcomes, p=transfer_probs)

        steps+=1
        
        #If we run out of steps, re-pull new dye coordinates
        if steps % n_frames == 0:
            dtrj = np.concatenate((
                dtrj,synthetic_trajectory(d_tprobs, dtrj[-1], n_frames)))
            atrj = np.concatenate((
                atrj,synthetic_trajectory(a_tprobs, atrj[-1], n_frames)))
            
    return([steps, d_state, dtrj[:steps], atrj[:steps]])