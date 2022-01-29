"""Model FRET dyes takes pdb structures and models dye pairs onto
a list of specified residue pairs. You can specify your own dyes or 
this will fall back onto the dyes, Alexa488 and Alexa594, that we supply
with Enspara. We return a probability distribution of dye distances of
length the number of structures provided.
"""

# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors: Justin J Miller <jjmiller@wustl.edu>
# Contributors: Louis Smith!
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential


import sys
import argpars
import os
import logging
import itertools
import inspect
import pickle
import json
from glob import glob
import subprocess as sp
from multiprocessing import Pool
from functools import partial
from enspara import ra
from enspara.geometry import dyes_from_expt_dist
from enspara.apps.util import readable_dir


import numpy as np
import mdtraj as md

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def process_command_line(argv):
##Need to check whether these arguments are in fact parsed correctly, I took a first pass stab at this.
##Better to make a flag that lets you stop after calculating FRET dye distributions?
    parser = argparse.ArgumentParser(
        prog='FRET',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert PDB structures a series of FRET dye residue pairs"
                    "into the probability distribution of distances between the two pairs")

#Add additional input to say "calc distributions, calc FEs, or both"
    # INPUTS
    input_args = parser.add_argument_group("Input Settings")
    input_args.add_argument(
        '--centers', nargs="+", required=True, 
        help="Path to cluster centers from the MSM"
             "should be of type .xtc. Not needed if supplying FRET dye distributions")
    input_args.add_argument(
        '--topology', required=True, action='append',
        help="topology file for supplied trajectory")
    input_args.add_argument(
        '--resid_pairs', nargs="+", action='append', required=True, type=int,
        help="residues to model FRET dyes on. Pass 2 residue pairs, same numbering as"
             "in the topology file. Pass multiple times to model multiple residue pairs"
             "Pass in the same order as your naming convention for dye distance distributions"
             "e.g. --resid_pairs 1 5"
             "--resid_pairs 5 86")


    # PARAMETERS
    FRET_args = parser.add_argument_group("FRET Settings")
    FRET_args.add_argument(
        '--n_procs', required=False, type=int, default=1,
        help="Number of cores to use for parallel processing"
            "Generally parallel over number of frames in supplied trajectory/MSM state")    
    FRET_args.add_argument(
        '--FRETdye1', nargs="+", required=False,
        default=os.path.dirname(inspect.getfile(ra))+'/../data/dyes/AF488.pdb',
        help="Path to point cloud of FRET dye pair 2")
    FRET_args.add_argument(
        '--FRETdye2', nargs = "+", required = False,
        default=os.path.dirname(inspect.getfile(ra))+'/../data/dyes/AF594.pdb',
        help = "Path to point cloud of FRET dye pair 2")

    # OUTPUT
    output_args = parser.add_argument_group("Output Settings")
    output_args.add_argument(
        '--FRET_dye_distributions', required=False, action=readable_dir,
        help="The location to write the FRET dye distributions.")
    output_args.add_argument(
        '--FRET_output_dir', required=False, action=readable_dir, default='./',
        help="The location to write the FRET dye distributions.")
    output_args.add_argument(
        '--FRET_output_names', required=True,
        help="Naming pattern for output FRET values.")

    args = parser.parse_args(argv[1:])
    #Need to add error checks?
    return args


def main(argv=None):

    args = process_command_line(argv)
    print(args)

    resSeq_pairs=np.array(args.resid_pairs)

    for n in np.arange(len(resSeq_pairs)):
        logger.info(f"Calculating distance distribution for residues {resSeq_pairs[n]}")
        probs, bin_edges = dyes.dye_distance_distribution(
            trj, AF488, AF594, resSeq_pairs[n], n_procs=n_procs)
        probs_output = '%s/probs_%s_%s.h5' % (base_name, resSeq_pairs[n][0], resSeq_pairs[n][1])
        bin_edges_output = '%s/bin_edges_%s_%s.h5' % (base_name, resSeq_pairs[n][0], resSeq_pairs[n][1])
        ra.save(probs_output, probs)
        ra.save(bin_edges_output, bin_edges)

    logger.info("Success! Calculated FRET dye distance distributions your input parameters can be found here: %s" % (output_folder + 'FRET_from_exp.json'))
    print(json.dumps(args.__dict__,  output_folder+'FRET_from_expt.json',indent=4))


if __name__ == "__main__":
    sys.exit(main(sys.argv))