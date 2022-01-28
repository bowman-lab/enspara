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
             "--resid_pairs 5 86"


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

    # Work greatly needed below! This is all just copy+paste from cluster.py's argparse
    # Ideally should have some checks (e.g. can't supply only centers.xtc and not top)
    args = parser.parse_args(argv[1:])

    #Minimum checks- 
    #see if need to do FRET_dye modeling, FE, or both
    #Check to make sure all arguments are provided
    #Return helpful errors?


    #
    #     if args.cluster_distance in FEATURE_DISTANCES:
    #         args.cluster_distance = getattr(libdist, args.cluster_distance)
    #     else:
    #         raise exception.ImproperlyConfigured(
    #             "The given distance (%s) is not compatible with features." %
    #             args.cluster_distance)
    #
    #     if args.subsample != 1 and len(args.features) == 1:
    #             raise exception.ImproperlyConfigured(
    #                 "Subsampling is not supported for h5 inputs.")
    #
    #     #TODO: not necessary if mutually exclusive above works
    #     if args.trajectories:
    #         raise exception.ImproperlyConfigured(
    #             "--features and --trajectories are mutually exclusive. "
    #             "Either trajectories or features, not both, are clustered.")
    #     if args.topologies:
    #         raise exception.ImproperlyConfigured(
    #             "When --features is specified, --topology is unneccessary.")
    #     if args.atoms:
    #         raise exception.ImproperlyConfigured(
    #             "Option --atoms is only meaningful when clustering "
    #             "trajectories.")
    #     if not args.cluster_distance:
    #         raise exception.ImproperlyConfigured(
    #             "Option --cluster-distance is required when clustering "
    #             "features.")
    #
    # elif args.trajectories and args.topologies:
    #     args.trajectories = expand_files(args.trajectories)
    #
    #     if not args.cluster_distance or args.cluster_distance == 'rmsd':
    #         args.cluster_distance = md.rmsd
    #     else:
    #         raise exception.ImproperlyConfigured(
    #             "Option --cluster-distance must be rmsd when clustering "
    #             "trajectories.")
    #
    #     if not args.atoms:
    #         raise exception.ImproperlyConfigured(
    #             "Option --atoms is required when clustering trajectories.")
    #     elif len(args.atoms) == 1:
    #         args.atoms = args.atoms * len(args.trajectories)
    #     elif len(args.atoms) != len(args.trajectories):
    #         raise exception.ImproperlyConfigured(
    #             "Flag --atoms must be provided either once (selection is "
    #             "applied to all trajectories) or the same number of times "
    #             "--trajectories is supplied.")
    #
    #     if len(args.topologies) != len(args.trajectories):
    #         raise exception.ImproperlyConfigured(
    #             "The number of --topology and --trajectory flags must agree.")
    #
    # else:
    #     # CANNOT CLUSTER
    #     raise exception.ImproperlyConfigured(
    #         "Either --features or both of --trajectories and --topologies "
    #         "are required.")
    #
    # if args.cluster_radius is None and args.cluster_number is None:
    #     raise exception.ImproperlyConfigured(
    #         "At least one of --cluster-radius and --cluster-number is "
    #         "required to cluster.")
    #
    # if args.algorithm == 'kcenters':
    #     args.Clusterer = KCenters
    #     if args.cluster_iterations is not None:
    #         raise exception.ImproperlyConfigured(
    #             "--cluster-iterations only has an effect when using an "
    #             "interative clustering scheme (e.g. khybrid).")
    # elif args.algorithm == 'khybrid':
    #     args.Clusterer = KHybrid
    #
    # if args.no_reassign and args.subsample == 1:
    #     logger.warn("When subsampling is 1 (or unspecified), "
    #                 "--no-reassign has no effect.")
    # if not args.no_reassign and mpi_mode and args.subsample > 1:
    #     logger.warn("Reassignment is suppressed in MPI mode.")
    #     args.no_reassign = True
    #
    # if args.trajectories:
    #     if os.path.splitext(args.center_features)[1] == '.h5':
    #         logger.warn(
    #             "You provided a centers file (%s) that looks like it's "
    #             "an h5... centers are saved as pickle. Are you sure this "
    #             "is what you want?")
    # else:
    #     if os.path.splitext(args.center_features)[1] != '.npy':
    #         logger.warn(
    #             "You provided a centers file (%s) that looks like it's not "
    #             "an npy, but this is how they are saved. Are you sure "
    #             "this is what you want?" %
    #             os.path.basename(args.center_features))

    return args


def main(argv=None):

    args = process_command_line(argv)
    print(args)
    # for n in np.arange(len(resSeq_pairs)):
    #     logger.info("Calculating distance distribution for residues %s", resSeq_pairs[n])
    #     probs, bin_edges = dyes.dye_distance_distribution(
    #         trj, AF488, AF594, resSeq_pairs[n], n_procs=n_procs)
    #     probs_output = '%s/probs_%sC_%sC.h5' % (base_name, resSeq_pairs[n][0], resSeq_pairs[n][1])
    #     bin_edges_output = '%s/bin_edges_%sC_%sC.h5' % (base_name, resSeq_pairs[n][0], resSeq_pairs[n][1])
    #     ra.save(probs_output, probs)
    #     ra.save(bin_edges_output, bin_edges)

    logger.info("Success! Calculated FRET dye distance distributions your input parameters can be found here: %s" % (output_folder + 'FRET_from_exp.json'))
    print(json.dumps(args.__dict__,  output_folder+'FRET_from_expt.json',indent=4))


if __name__ == "__main__":
    sys.exit(main(sys.argv))