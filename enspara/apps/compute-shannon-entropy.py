# -*- coding: utf-8 -*-

"""This apps script computes the Shannon entropy for each residue using the definition
of rotamers established by the CARDS framework. Shannon entropy is computed for each 
individual dihedral before being combined on a per-residue basis. A normalization is 
also applied so that the per-residue entropy spans from 0 to 1, where 1 is the 
maximum possible entropy for a single residue. Each dihedral has either 2 or 3 rotameric 
states, for backbone and sidechain dihedrals respectively. 

If you use this Shannon entropy, please cite the following paper: 
-----------------------------------------------------
[1] Sukrit Singh and Gregory R. Bowman, "Quantifying allosteric communication via 
    both concerted structural changes and conformational disorder with CARDS".
    Journal of Chemical Theory and Computation 2017 13 (4), 1509-1517
    DOI: 10.1021/acs.jctc.6b01181 

[2] Justin R Porter, Maxwell I Zimmerman, Gregory R Bowman, "Enspara: Modeling molecular 
    ensembles with scalable data structures and parallel computing". 
    bioRxiv 431072; doi: https://doi.org/10.1101/431072 
"""

import sys
import argparse
import os
import logging
import itertools
import pickle
import json
import warnings
import numpy as np
import mdtraj as md

from glob import glob 
from enspara.cards import cards
from enspara.util.parallel import auto_nprocs
from enspara.util import array as ra
from enspara.util import load_as_concatenated
from enspara.apps.util import readable_dir
from enspara.util.log import timed
from enspara.cards import featurizers as feat
from enspara.info_theory import entropy as ent
from enspara.info_theory import mutual_info as mut

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s %(name)-8s %(levelname)-7s %(message)s'),
    datefmt='%m-%d-%Y %H:%M:%S')

from enspara.geometry import libdist

from enspara import exception

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def process_command_line(argv):
    """Parse the command line and do a first-pass on processing them into a
    format appropriate for the rest of the script.

    Parameters
    ----------
    argv : list of str
        The command line arguments.

    Returns
    -------
    args : argparse.Namespace
        The parsed arguments.

    Raises
    ------
    exception.ImproperlyConfigured
        If the buffer size is not between 0 and 360.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Compute Shannon entropy per dihedral for a set of trajectories "
                    "and save entropies and dihedral mappings.\n \n"
                    "Please cite the following papers if you use CARDS with enspara:\n"
                    "[1] Singh, S. and Bowman, G.R.\n" 
                    "    Journal of Chemical Theory and Computation\n"
                    "    2017 13 (4), 1509-1517\n"
                    "    DOI: 10.1021/acs.jctc.6b01181\n"
                    "\n"
                    "[2] Porter,J.R.,  Zimmerman, M.I., and Bowman G.R.\n"
                    "    bioRxiv 431072; doi: https://doi.org/10.1101/431072\n")

    # INPUTS
    input_args = parser.add_argument_group("Input Settings")
    input_args.add_argument(
        '--trajectories', required=True, nargs="+", action='append',
        help="List of paths to aligned trajectory files to cluster. "
             "All file types that MDTraj supports are supported here.")
    input_args.add_argument(
        '--topology', required=True, action='append',
        help="The topology file for the trajectories.")

    # PARAMETERS
    cards_args = parser.add_argument_group("CARDS Settings")
    cards_args.add_argument(
        '--buffer-size', default=15, type=int,
        help="Size of buffer zone between rotameric states, in degrees.")
    cards_args.add_argument(
        "--processes", default=max(1, auto_nprocs()/4), type=int,
        help="Number of processes to use.")

    # OUTPUT
    output_args = parser.add_argument_group("Output Settings")
    output_args.add_argument(
        '--entropies', action=readable_dir,
        help="The location to write the normalized entropies file (as CSV)")

    args = parser.parse_args(argv[1:])

    # FEATURES
    if not (0 < args.buffer_size < 360):
        raise exception.ImproperlyConfigured(
            "The given buffer size (%s) is not possible." %
            args.buffer_size)

    return args


def load_trajs(args):
    """ Creates a generator object that is passed onto the CARDS framework.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    gen : generator
        A generator object that yields the loaded trajectories.

    Raises
    ------
    exception.ImproperlyConfigured
        If the number of trajectories and topologies do not match.
    """
    trajectories = args.trajectories
    topology = args.topology[0]
    #filenames = glob(trajectories)
    targets = {os.path.basename(topf): "%s files" % len(trjfs) for topf, trjfs
               in zip(args.topology, args.trajectories)}
    logger.info("Computing Shannon entropies; targets:\n%s",
                json.dumps(targets, indent=4))

    gen = (md.load(traj, top=topology) for traj in args.trajectories)

    return gen


def compute_rotamer_counts(rotamers): 
    """Use existing framework of computing joint counts matrices
    to compute the rotamer counts for each dihedral across each trajectory.

    Parameters
    ----------
    rotamers : RotamerFeaturizer
        The RotamerFeaturizer object from the CARDS framework.

    Returns
    -------
    final_counts : np.array
        The final counts for each dihedral across all trajectories.
    """
    jc = None
    feature_trajs = rotamers.feature_trajectories_
    num_rotamer_features = rotamers.n_feature_states_

    for i,(x,y) in enumerate(zip(feature_trajs, feature_trajs)):
        jc_i = mut.joint_counts(x,y, np.max(num_rotamer_features), 
                                np.max(num_rotamer_features))

        if not hasattr(jc, 'shape'):
            jc = jc_i
        else: 
            jc += jc_i

    # The final jc matrix represents the joint counts matrix for each rotamer. 
    # We can conver this joint counts matrix into a set of counts per matrix
    # this can be done by summing across each joint_count matrix
    n_obs_a_i = jc.sum(axis=-1)

    # However, this amount of data is redundant, since each element at [i,i] 
    # will contain the counts we need to compute entropies
    # This can be done relatively easily
    final_counts = []
    for i in range(jc.shape[0]):
        counts = n_obs_a_i[i,i]
        final_counts.append(counts)

    return np.asarray(final_counts)

def compute_dihedral_shannon_entropy(probs):
    """Computes a shannon entropy for every dihedral in the simulation set. 

    Parameters
    ----------
    probs : np.array
        The probability distribution for each dihedral.

    Returns
    -------
    entropy_values : np.array
        The entropy values for each dihedral.
    """
    num_dihedrals = probs.shape[0]

    entropy_values = np.zeros(shape=num_dihedrals)

    for i in range(num_dihedrals):
        entropy_values[i] = ent.shannon_entropy(probs[i])

    return entropy_values


def sum_dihedral_entropies(dihedral_entropies, resi_mapping, n_resis):
    """This sums the dihedral entropies into an array of per-residue entropies.

    Parameters
    ----------
    dihedral_entropies : np.array
        The entropies for each dihedral.
    resi_mapping : np.array
        The mapping of each dihedral to a residue.
    n_resis : int
        The number of residues in the system.

    Returns
    -------
    summed_entropies : np.array
        The summed entropies for each residue.
    """
    summed_entropies = np.zeros(n_resis)
    for i in range(n_resis):
        summed_entropies[i] = dihedral_entropies[resi_mapping == i].sum()

    return summed_entropies

def compute_channel_capacities(n_states_array, resi_list, n_resis):
    """This computes the maximum possible entropy any one residue can have, based on 
    the number of dihedrals it has and the number of states each dihedral can adopt.

    Parameters
    ----------
    n_states_array : np.array
        The number of states for each dihedral.
    resi_list : np.array
        The mapping of each dihedral to a residue.
    n_resis : int
        The number of residues in the system.
    """
    # The maximum possible entropy for any one residue is 
    # np.sum(n*log(b)) where there are n total dihedrals and each dihedral has b states
    # so this sums across all n and all b 

    channel_capacities = np.zeros(n_resis)

    for i in range(n_resis):
        rots_per_residue = n_states_array[resi_list == i]
        channel_capacities[i] = np.sum([np.sum(np.log(val)) 
            for val in rots_per_residue])

    return channel_capacities


def compute_residue_shannon_entropies(
    dihedral_entropies, topologyFile, atom_inds, n_states):
    """Compiles the dihedral level entropies into a single list of per-residue 
    Shannon entropies. Returns both as separate arrays.

    Parameters
    ----------
    dihedral_entropies : np.array
        The entropies for each dihedral.
    topologyFile : str
        The topology file for the system.
    atom_inds : np.array
        The atom indices for each dihedral.
    n_states : np.array
        The number of states for each dihedral.

    Returns
    -------
    normalized_entropies : np.array
        The normalized Shannon entropies for each residue.  
    resi_list : np.array
        The list of unique residues in the system.
    """

    # First we need to load our topology for matching up residues
    structure = md.load(topologyFile)
    n_resis = structure.top.n_residues
    num_dihedrals = dihedral_entropies.shape[0]

    # Now we define a mapping array to identify the residue each entropy maps to
    resi_list = np.zeros(num_dihedrals)
    
    # Now we identify which residue each dihedral belongs to
    for i in range(num_dihedrals):
        dihedral = atom_inds[i]
        identifying_atom = dihedral[1]
        # Subtract 1 from the index we extract because residue numbering starts at 1
        index_val = structure.top.atom(identifying_atom).residue.resSeq - 1 
        resi_list[i] = index_val

    # now that we've populated the mapping - let's combine some dihedrals and normalize
    # First we compute the total entroy per residue
    total_entropies = sum_dihedral_entropies(dihedral_entropies, resi_list, 
                                                structure.top.n_residues)
    # Then we compute each residue's total possible entropy 
    residue_channel_capacity = compute_channel_capacities(n_states, resi_list, 
                                                            structure.top.n_residues)

    # Finally, we do total/capacity to normalize - this is our final array
    normalized_entropies = total_entropies / residue_channel_capacity
    for i in range(n_resis):
        if total_entropies[i] > residue_channel_capacity[i]: print("bug")

    # We will return the final normalized entropies and a simple list of unique
    # residue IDs - making it easier to manage

    # We add one back to the resi_list we return because it is not used for indexing but
    # for saving on a per residue basis

    return normalized_entropies, np.unique(resi_list+1) 


def compute_shannon_entropies(args, trj_list):
    """Main modular method which takes arguments and computes the final per-residue
    Shannon entropy for saving.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.
    trj_list : generator
        A generator object that yields the loaded trajectories.

    Returns
    -------
    residue_entropy : np.array
        The final per-residue entropies.
    resi_list : np.array
        The list of unique residues in the system.
    """
    trajectories = args.trajectories
    topology = args.topology[0]

    # First we extract the rotamers using the Rotamer_Featurizer within CARDS
    rotamers = feat.RotamerFeaturizer(args.buffer_size, args.processes)
    rotamers.fit(trj_list)

    # Then we convert the counts for each dihedral's rotamers
    counts = compute_rotamer_counts(rotamers)

    
    # Now that we have the counts per dihedral, we convert them  
    # probabilities for each rotameric bin
    P_a = counts/counts.sum(axis=-1)[...,None]

    # P_a now contains the probability distribution across rotamers for each 
    # dihedral. Each dihedral represents a single row in P_a

    # From these probabilities, we can compute the total shannon entropy for 
    # each dihedal 
    entropy_per_dihedral = compute_dihedral_shannon_entropy(P_a)

    # Now we need to combine these entropies on a per-residue basis
    # When we do so, we all need to normalize out each per-residue entropy by 
    # the maximum possible entropy of that residue (or the channel capacity)
    residue_entropy, resi_list = compute_residue_shannon_entropies(entropy_per_dihedral, 
                                                                    topology, 
                                                                    rotamers.atom_indices_,
                                                                    rotamers.n_feature_states_)
    return residue_entropy, resi_list


def save_all_entropies(entropies, residues, fileName):
    """Saves the final per-residue entropies as a CSV file with the corresponding
    residue ID. 

    Parameters
    ----------
    entropies : np.array
        The per-residue entropies.
    residues : np.array
        The list of unique residues in the system.
    fileName : str
        The name of the file to save the entropies to.

    Returns
    -------
    int
        0 if the file was saved successfully.
    """
    final_entropy_data = np.vstack((residues, entropies)).T

    np.savetxt(fileName, final_entropy_data, delimiter=",")

    return 0 



def main(argv=None):
    """Run the driver script for this module. This code only runs if we're
    being run as a script. Otherwise, it's silent and just exposes methods.
    
    Parameters
    ----------
    argv : list of str
        The command line arguments.
        
    Returns
    -------
    int
        The return code.

    Raises
    ------
    exception.ImproperlyConfigured
        If the buffer size is not between 0 and 360.
    """
    args = process_command_line(argv)
    
    logger.info("Loading trajectories...")
    trj_list = load_trajs(args)

    with timed("Calculating Shannon entropy took %.1f s.", logger.info):
        entropies, residues = compute_shannon_entropies(args, trj_list)

    logger.info("Completed entropy calculation. ")

    save_all_entropies(entropies, residues, args.entropies)

    logger.info("Saved all entropies as as %s", args.entropies)

    return 0 



if __name__ == "__main__":
    sys.exit(main(sys.argv))

