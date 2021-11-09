"""The smFRET app allows you to convert your MSM into a single-molecule
FRET histogram based on the residue pairs of interest. Parameters such 
as the dye identities and dye positions must be specified. This code also
enables adaptation to specific smFRET experimental setups enabling users 
to modify the number of bursts observed, distribution of photon arrival
times, and minimum number of binned photons. The app will return a list of 
FRET efficiencies that is n_bursts long. See the apps tab for more information.
"""

import sys
import argparse
import os
import logging
import itertools
import pickle
import json
from glob import glob

import numpy as np
import mdtraj as md


#EXPECTED inputs: 
# list of dye pairs
# n_photon_bursts
# min_photons_in_burst
# n_chunks (optional, default to false)
# photon_time
# eq_probs from MSM
# t_probs from MSM
# lagtime from MSM
# n_procs (defualt to 1)




def process_command_line(argv):

    parser = argparse.ArgumentParser(
        prog='FRET',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert an MSM and a series of FRET dye residue pairs"
                    "into the predicted FRET efficiency for each pair")

    # INPUTS
    input_args = parser.add_argument_group("Input Settings")
    input_data_group.add_argument(
        '--eq_probs', nargs="+", action='append',
        help="equilibrium probabilities from the MSM"
             "Should be of file type .npy")
        input_data_group.add_argument(
        '--t_probs', nargs="+", action='append',
        help="transition probabilities from the MSM"
             "Should be of file type .npy")
        input_data_group.add_argument(
        '--lagtime', nargs="+", action='append',
        help="lag time used to construct the MSM (in ns)"
             "Should be type float or int")        
        input_data_group.add_argument(
        '--resid_pairs', nargs="+", action='append',
        help="list of lists of residue pairs to sample, 1 indexed"
             "e.g.: [[[1],[2]],[[2],[52]]]")            
 

    # PARAMETERS
    FRET_args = parser.add_argument_group("FRET Settings")
    cluster_args.add_argument(
        '--n_photon_bursts', required=True, type=int,
        help="Number of photon bursts to observe in total")
    cluster_args.add_argument(
        '--min_photons', required=True, type=int,
        help="Minimum number of photons in a photon burst")
    cluster_args.add_argument(
        '--n_chunks', required=False, type=int, default=0
        help="Enables you to assess intraburst variation."
        	"How many chunks would you like a given burst broken into?")
    cluster_args.add_argument(
        '--photon_time', required=False, type=int, default=4
        help="This defines the distribution of photon arrival times"
        	"Currently implemented using np.random.exponential with scale of photon_time")
    cluster_args.add_argument(
        '--n_procs', required=False, type=int, default=1
        help="Number of cores to use for parallel processing"
        	"Generally parallel over number of frames in supplied trajectory/MSM state")
    cluster_args.add_argument(
        '--trj', nargs="+", required=False action=readable_dir,
        help="Path to cluster centers from the MSM"
             "should be of type .xtc. Not needed if supplying FRET dye distributions")       
    cluster_args.add_argument(
        '--FRET_dye_dist', nargs="+", required=False, action=readable_dir,
        help="Path to FRET dye distributions")

    # OUTPUT
    output_args = parser.add_argument_group("Output Settings")

    output_args.add_argument(
        '--FRET_dye_distributions', required=False, action=readable_dir,
        help="The location to write the FRET dye distributions.")
    output_args.add_argument(
        '--FRET_efficiencies', required=True, action=readable_dir,
        help="The location to write the predicted FRET efficiencies for each residue pair.")


    args = parser.parse_args(argv[1:])

    if args.features:
        args.features = expand_files([args.features])[0]

        if args.cluster_distance in FEATURE_DISTANCES:
            args.cluster_distance = getattr(libdist, args.cluster_distance)
        else:
            raise exception.ImproperlyConfigured(
                "The given distance (%s) is not compatible with features." %
                args.cluster_distance)

        if args.subsample != 1 and len(args.features) == 1:
                raise exception.ImproperlyConfigured(
                    "Subsampling is not supported for h5 inputs.")

        # TODO: not necessary if mutually exclusvie above works
        if args.trajectories:
            raise exception.ImproperlyConfigured(
                "--features and --trajectories are mutually exclusive. "
                "Either trajectories or features, not both, are clustered.")
        if args.topologies:
            raise exception.ImproperlyConfigured(
                "When --features is specified, --topology is unneccessary.")
        if args.atoms:
            raise exception.ImproperlyConfigured(
                "Option --atoms is only meaningful when clustering "
                "trajectories.")
        if not args.cluster_distance:
            raise exception.ImproperlyConfigured(
                "Option --cluster-distance is required when clustering "
                "features.")

    elif args.trajectories and args.topologies:
        args.trajectories = expand_files(args.trajectories)

        if not args.cluster_distance or args.cluster_distance == 'rmsd':
            args.cluster_distance = md.rmsd
        else:
            raise exception.ImproperlyConfigured(
                "Option --cluster-distance must be rmsd when clustering "
                "trajectories.")

        if not args.atoms:
            raise exception.ImproperlyConfigured(
                "Option --atoms is required when clustering trajectories.")
        elif len(args.atoms) == 1:
            args.atoms = args.atoms * len(args.trajectories)
        elif len(args.atoms) != len(args.trajectories):
            raise exception.ImproperlyConfigured(
                "Flag --atoms must be provided either once (selection is "
                "applied to all trajectories) or the same number of times "
                "--trajectories is supplied.")

        if len(args.topologies) != len(args.trajectories):
            raise exception.ImproperlyConfigured(
                "The number of --topology and --trajectory flags must agree.")

    else:
        # CANNOT CLUSTER
        raise exception.ImproperlyConfigured(
            "Either --features or both of --trajectories and --topologies "
            "are required.")

    if args.cluster_radius is None and args.cluster_number is None:
        raise exception.ImproperlyConfigured(
            "At least one of --cluster-radius and --cluster-number is "
            "required to cluster.")

    if args.algorithm == 'kcenters':
        args.Clusterer = KCenters
        if args.cluster_iterations is not None:
            raise exception.ImproperlyConfigured(
                "--cluster-iterations only has an effect when using an "
                "interative clustering scheme (e.g. khybrid).")
    elif args.algorithm == 'khybrid':
        args.Clusterer = KHybrid

    if args.no_reassign and args.subsample == 1:
        logger.warn("When subsampling is 1 (or unspecified), "
                    "--no-reassign has no effect.")
    if not args.no_reassign and mpi_mode and args.subsample > 1:
        logger.warn("Reassignment is suppressed in MPI mode.")
        args.no_reassign = True

    if args.trajectories:
        if os.path.splitext(args.center_features)[1] == '.h5':
            logger.warn(
                "You provided a centers file (%s) that looks like it's "
                "an h5... centers are saved as pickle. Are you sure this "
                "is what you want?")
    else:
        if os.path.splitext(args.center_features)[1] != '.npy':
            logger.warn(
                "You provided a centers file (%s) that looks like it's not "
                "an npy, but this is how they are saved. Are you sure "
                "this is what you want?" %
                os.path.basename(args.center_features))

    return args