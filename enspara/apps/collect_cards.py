# -*- coding: utf-8 -*-

"""CARDS is a method for quantifying correlated motions between residues in a 
protein. This method works by classifying all dihedrals of a protein into 
rotameric and dynamical states. Each dihedral has either 2 or 3 rotameric 
states, for backbone and sidechain dihedrals respectively, and 2 dynamical states 
representing whether or not the dihedral is ordered or disordered. 
If you use CARDS, please cite the following papers: 
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


logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s %(name)-8s %(levelname)-7s %(message)s'),
    datefmt='%m-%d-%Y %H:%M:%S')


from enspara.geometry import libdist

from enspara import exception


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_command_line(argv):
    '''Parse the command line and do a first-pass on processing them into a
    format appropriate for the rest of the script.'''

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Compute CARDS matricies for a set of trajectories "
                    "and save all matrices and dihedral mappings.\n \n"
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
    #input_data_group = parser.add_mutually_exclusive_group(required=True)
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
        '--matrices', required=True, action=readable_dir,
        help="The folder location to write the four CARDS matrices (as pickle).")
    output_args.add_argument(
        '--indices', required=True, action=readable_dir,
        help="The location to write the dihedral indices file (as CSV).")

    args = parser.parse_args(argv[1:])

    # CARDS FEATURES
    if not (0 < args.buffer_size < 360):
        raise exception.ImproperlyConfigured(
            "The given buffer size (%s) is not possible." %
            args.buffer_size)

    return args



def load_trajectory_generator(trajectories, topology):

    for i,t in enumerate(trajectories):
        logger.info('loading '+str(t))
        yield md.load(t, top=topology)



def load_trajs(args):
    """ Creates a generator object that can be then passed to the CARDS framework.
    """
    trajectories = args.trajectories[0]
    topology = args.topology[0]
    #filenames = glob(trajectories)
    targets = {os.path.basename(topf): "%s files" % len(trjfs) for topf, trjfs
               in zip(args.topology, args.trajectories)}
    logger.info("Starting CARDS; targets:\n%s",
                json.dumps(targets, indent=4))

    #gen = (md.load(traj, top=topology) for traj in args.trajectories)
    gen = load_trajectory_generator(trajectories, topology)

    logger.info("Created generator")

    return gen


def save_cards(ss_mi, dd_mi, sd_mi, ds_mi, outputName):
    """Save the four cards matrices as a single pickle file
    """

    #final_mats = [ss_mi, dd_mi, sd_mi, ds_mi]
    final_mats = {
        'Struc_struc_MI': ss_mi, 
        'Disorder_disorder_MI': dd_mi,
        'Struc_disorder_MI': sd_mi,
        'Disorder_struc_MI': ds_mi, }
    
    logger.info("Saving matrices - saved as %s", outputName)

    with open(outputName, 'wb') as f:
        pickle.dump(final_mats, f)

    return 0 




def main(argv=None):
    """Run the driver script for this module. This code only runs if we're
    being run as a script. Otherwise, it's silent and just exposes methods.
    """
    args = process_command_line(argv)

    trj_list = load_trajs(args)

    with timed("Calculating CARDS correlations took %.1f s.", logger.info):
        ss_mi, dd_mi, sd_mi, ds_mi, inds = cards(trj_list, args.buffer_size, 
                                                        args.processes)

    logger.info("Completed correlations. ")

    save_cards(ss_mi, dd_mi, sd_mi, ds_mi, args.matrices)
    np.savetxt(args.indices, inds, delimiter=",")

    logger.info("Saved dihedral indices as %s", args.indices)

    return 0 



if __name__ == "__main__":
    sys.exit(main(sys.argv))
