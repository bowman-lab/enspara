"""The smFRET app allows you to convert your MSM into a single-molecule
FRET histogram based on the residue pairs of interest. Users specify
MSM structure centers and the transition probabilities between each center.
Parameters such as the dye identities can be changed if you have your own point clouds.
Dyes may be mapped to any amino acid in the protein. As single-molecule FRET is 
highly dependent on true-to-experiment simulation timescales, you can also rescale the 
speed of your trajectories within an MSM. This code also enables adaptation to specific
smFRET experimental setups as users provide their own experimental FRET bursts. 
See the apps tab for more information.
"""
# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors: Justin J Miller <jjmiller@wustl.edu>
# Contributors: Louis Smith!
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential


import sys
import argparse
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
        description="Convert an MSM and a series of FRET dye residue pairs"
                    "into the predicted FRET efficiency for each pair")

#Add additional input to say "calc distributions, calc FEs, or both"
    # INPUTS
    input_args = parser.add_argument_group("Input Settings")
    input_args.add_argument(
        '--eq_probs', required= True,
        help="equilibrium probabilities from the MSM"
             "Should be of file type .npy")
    input_args.add_argument(
        '--t_probs', required=True,
        help="transition probabilities from the MSM"
             "Should be of file type .npy")
    input_args.add_argument(
        '--photon_times', required=True,
        help="List of lists of photon arrival times."
             "each list is an individual photon burst"
             "with cumulative arrival times (in us) for each burst"
             "Should be of file type .npy")    
    input_args.add_argument(
        '--lagtime', type=int, required=True,
        help="lag time used to construct the MSM (in ns)"
             "Should be type float")
    input_args.add_argument(
        '--resid_pairs', nargs="+", action='append', required=True, type=int,
        help="residues to model FRET dyes on. Pass 2 residue pairs, same numbering as"
             "in the topology file. Pass multiple times to model multiple residue pairs"
             "e.g. --resid_pairs 1 5"
             "--resid_pairs 5 86")
    input_args.add_argument(
        '--FRET_dye_dists', nargs="+", required=False, action=readable_dir,
        help="Path to FRET dye distributions")    


    # PARAMETERS
    FRET_args = parser.add_argument_group("FRET Settings")
    FRET_args.add_argument(
        '--n_procs', required=False, type=int, default=1,
        help="Number of cores to use for parallel processing"
            "Generally parallel over number of frames in supplied trajectory/MSM state")    
    FRET_args.add_argument(
        '--n_chunks', required=False, type=int, default=0,
        help="Enables you to assess intraburst variation."
        	"How many chunks would you like a given burst broken into?")
    FRET_args.add_argument(
        '--R0', nargs="+", required=False, type=float, default=5.4,
        help="R0 value for FRET dye pair of interest")
    FRET_args.add_argument(
        '--slowing_factor', required=False, type=int, default=1,
        help= "factor to slow your trajcetories by")

    # OUTPUT
    output_args = parser.add_argument_group("Output Settings")

    output_args.add_argument(
        '--FRET_output_dir', required=False, action=readable_dir, default='./',
        help="The location to write the FRET dye distributions.")

    args = parser.parse_args(argv[1:])
    #Add error checkers?? None come to mind at the moment...
    return args


def main(argv=None):

    args = process_command_line(argv)

    #Load necessary data
    t_probabilties= np.load(args.t_probs)
    logger.info(f"Loaded t_probs from {args.t_probs}")
    populations=np.load(args.eq_probs)
    logger.info(f"Loaded eq_probs from {args.eq_probs}")
    resSeq_pairs=np.array(args.resid_pairs)
    cumulative_times=np.load(args.photon_times)
    
    #Convert Photon arrival times into MSM steps.
    MSM_frames=dyes_from_expt_dist(cumulative_times, args.lagtime, args.slowing_factor)

    #Calculate the FRET efficiencies
    for n in np.arange(resSeq_pairs.shape[0]):
        title = f'{resSeq_pairs[n,0]}_{resSeq_pairs[n,1]}'
        probs_file = f"{args.FRET_dye_dists}/probs_{title}.h5"
        bin_edges_file = f"{args.FRET_dye_dists}/bin_edges_{title}.h5"
        probs = ra.load(probs_file)
        bin_edges = ra.load(bin_edges_file)
        dist_distribution = dyes_from_expt_dist.make_distribution(probs, bin_edges)
        FEs_sampling = dyes_from_expt_dist.sample_FRET_histograms(
            T=t_probabilties, populations=populations, dist_distribution=dist_distribution,
            MSM_frames=MSM_frames, n_photon_std=args.n_chunks, n_procs=args.n_procs, R0=args.R0)
        np.save(f"{FRET_output_dir}/{FRET_output_names}_{title}_time_factor_{args.slowing_factor}.npy", FEs_sampling)


    logger.info(f"Success! Your FRET data can be found here: {FRET_output_dir}"
    # print(json.dumps(args.__dict__,  output_folder+'FRET_inputs.json',indent=4))


if __name__ == "__main__":
    sys.exit(main(sys.argv))