import itertools
import mdtraj as md
import numpy as np
import sys
from pylab import *

def get_cdf(data, pops):
    """Returns a CDF based on data and populations"""
    pops = pops/pops.sum()
    iis = np.argsort(data)
    ys = np.cumsum(pops[iis])
    return data[iis], ys

def average_chunks(data, avg_num=1, axis=0):
    """Averages numbers in a list in the following way:
    [1, 1, 1, 2, 2, 2, 3, 3, 3] -> [1_avg, 2_avg, 3_avg]"""
    avg_chunks = np.sum(
        np.array(
            [datas[num::avg_num] for num in range(avg_num)]), axis=0)/avg_num
    return avg_chunks

def return_state_iis(states, states_subset):
    """Returns index location of states_subset in states"""
    return np.array(
       [np.where(states==num)[0][0] for num in states_subset])

def compute_conditional_cdfs(
        values, populations, conditional_states=None, orig_states=None):
    """Return cumulative distribution functions (CDFs) based on
    populations.

    Parameters
    ----------
    values : array, shape [n_states, n_cdfs]
        The values used to compute a CDF. Each column vector represents
        a list of values to generate a CDF, and the number of columns
        will be the number of CDFs computed.
    populations : array, shape [n_states, ]
        The populations of each state.
    conditional_states : array, shape [m_states, ], optional, default: None
        A list of states to use for generation of a CDF.
    orig_states : array, shape [n_states, ], optional, default: None
        A list of the state numberings that correspond to each
        population. This is necessary if conditional states are used
        where numberings are pulled from a list of states that is not
        sequential

    Returns
    ----------
    cdfs : array, shape [n_cdfs, 2, n_data_points]
        Each row of cdfs is an array of the x and y values of the
        calculated cdf.
    """
    # get proper state numberings
    if orig_states is None:
        states = np.where(populations>0)[0]
    else:
        states = orig_states
    # index numbers of conditional states
    if conditional_states is None:
        conditional_states = states
        conditional_state_iis = np.arange(len(states))
    else:
        conditional_state_iis = return_state_iis(states, conditional_states)
    # get cdfs
    cdfs = []
    if values.ndim > 1:
        conditional_pops = populations[conditional_state_iis]
        conditional_values = values[conditional_state_iis]
        for num in range(len(values[0])):
            cdf = get_cdf(conditional_values[:, num], conditional_pops)
            cdfs.append(cdf)
    else:
        conditional_pops = popuations[conditional_state_iis]
        conditional_values = values[conditional_state_iis]
        cdfs = [get_cdf(conditional_values, conditional_pops)]
    return cdfs

def _convert_atom_names(top, apairs):
    """A very specific function that converts atom names into atom
    indices, when present.

    Parameters
    ----------
    top : mdtraj topology object
    apairs : array, shape [n_pairs, 2, n_equivalent]
        see compute_trj_distances.

    Returns
    ----------
    new_apairs : array, shape [n_pairs, 2, n_equivalent]
        The same array as apairs, except string atom-names
        are converted into index numbers.
    """
    # Make a copy of apairs
    new_apairs = np.array(apairs, copy=True)
    # Iterate over all elements and convert when necessary
    for dist_pair_ii in range(len(apairs)):
        dist_pair = apairs[dist_pair_ii]
        equiv_atoms_range = range(len(apairs[dist_pair_ii]))
        if len(equiv_atoms_range) != 2:
            raise # improperly configued
        for equiv_atoms_ii in equiv_atoms_range:
            equiv_atoms = dist_pair[equiv_atoms_ii]
            atom_index_range = range(len(apairs[dist_pair_ii][equiv_atoms_ii]))
            for atom_index_ii in atom_index_range:
                atom_to_test = equiv_atoms[atom_index_ii]
                # test if neceaary to convert
                if type(atom_to_test) is int:
                    pass
                elif type(atom_to_test) is str:
                    resi = atom_to_test.split('-')[0][3:]
                    atomname = atom_to_test.split('-')[1]
                    atom_index = top.select(
                        "resSeq "+str(resi)+" and name "+atomname)
                    # If len of mdtraj return is not equal to one, the atom
                    # was not found or the selection is wrong
                    if len(atom_index) != 1:
                        raise # atom not found!
                    else:
                        atom_index = atom_index[0]
                    new_apairs[dist_pair_ii][equiv_atoms_ii][atom_index_ii] = \
                        atom_index
                else:
                    raise # improperly configued
    return new_apairs

def compute_trj_distances(trj, apairs, min_dist=True):
    """Computes distances between pairs of atoms in a trajectory

    Parameters
    ----------
    trj : md.trajectory object
        An MDTraj trajectoroy containing atom-positions and
        topology information
    apairs : array, shape [n_pairs, 2, n_equivalent]
        An array of atom-pairs to calculate distances from. Atom-pairs
        can be either atom-indices or atom-names. If multiple indices
        or names are supplied within a pair, they will be treated as
        equivalent atoms. For example:

        >>> apairs = [
            [[150], [160]],
            [['GLU64-OE1', 'GLU64-OE2'], ['ASN20-HD21', 'ASN20-HD22']],
            ]

        will return the distance between atom-indices 150-160 and the
        minimum distance between GLU64's OD1/OD2 to ASN182's HD21/HD22.
        Here, each pair, OD1/OD2 and HD21/HD22, are treated as
        equivalent atoms, respectively.
    min_dist : boolean, optional, default: True
        Optionally reports the minimum distance between equivalent
        atoms. If False, will report the maximum distance.

    Returns
    ----------
    dists : array, shape [trj_length, n_pairs]
        Distances between specified pairs of atoms
    """
    top = trj.topology
    apairs = _convert_atom_names(top, apairs)
    dists = []
    for num in range(len(apairs)):
        dpairs = itertools.product(apairs[num][0], apairs[num][1])
        dists_tmp = md.compute_distances(trj, atom_pairs=dpairs)
        if min_dist:
            dists.append(dists_tmp.min(axis=1))
        else:
            dists.append(dists_tmp.max(axis=1))
    return np.array(dists).T

def conditional_dist_plots(
        trj, populations, apairs, conditional_states=None, orig_states=None,
        min_dist=True, confs_per_state=1):
    """Generates cdfs of apair distances.

    Parameters
    ----------
    trj : md.trajectory object, shape [n_state * n_confs]
        An MDTraj trajectoroy containing atom-positions and
        topology information. This is thought to be a list of
        representative cluster centers to analyze specific distances.
        frames are assumed to be in sequential state ordering with
        multiple representative conformations of a single state.
        Each state must have the same number of conformations i.e.
        for N states with 3 confs per state, the pdb-trajectory would
        look like the following:

        [state0-0, state0-1, state0-2, ... stateN-0, stateN-1, stateN-2]

    populations : array, shape [n_states, ]
        The populations of each state in sequential order.
    apairs : array, shape [n_pairs, 2, n_equivalent]
        see compute_trj_distances.
    conditional_states : array, shape [m_states, ], optional, default: None
        A list of states to use for generation of a CDF.
    orig_states : array, shape [n_states, ], optional, default: None
        A list of the state numberings that correspond to each
        population. This is necessary if conditional states are used
        where numberings are pulled from a list of states that is not
        sequential
    min_dist : boolean, optional, default: True
        Optionally reports the minimum distance between equivalent
        atoms. If False, will report the maximum distance.
    confs_per_state : int, optional, default: 1
        The number of conformations per state supplied in trj. If
        multiple conformations are supplied in the trajectory,
        distances will be averaged within each state before using
        population weightings.
    """
    dists = average_chunks(
        compute_trj_distances(trj, apairs, min_dist=min_dist),
        avg_num=confs_per_state, axis=0)
    cdfs = compute_conditional_cdfs(
        dists, populations, conditional_states=conditional_states,
        orig_states=orig_states)
    return cdfs

