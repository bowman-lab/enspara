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
# Contributors: Louis Smith <louissmith@wustl.edu>


import sys
import argparse
import logging
import os
import inspect
import re
from enspara import ra
import mdtraj as md
from enspara.geometry import dyes_from_expt_dist
from enspara.apps.util import readable_dir
import glob
from scipy.stats import entropy

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_command_line(argv):
    parser = argparse.ArgumentParser(
        prog='smFRET',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert an MSM and a series of FRET dye residue pairs into"
                    "the predicted FRET efficiency for each pair and fit to expt time"
                    "Multi-step process- model dyes, calc FRET efficiency, fit time factor")

    subparsers = parser.add_subparsers(title='commands', dest='command',
                                       description='valid subcommands',
                                       help='Call a subparser!')

    ############################
    ### Model_dyes subparser ###
    ############################
    model_dyes_parser = subparsers.add_parser('model_dyes', help='model FRET dyes onto MSM centers')

    # Model dyes INPUTS
    model_input_args = model_dyes_parser.add_argument_group("Input Settings")
    model_input_args.add_argument(
        'centers',
        help="Path to cluster centers from the MSM"
             "should be of type .xtc.")
    model_input_args.add_argument(
        'topology',
        help="topology file for supplied trajectory")
    model_input_args.add_argument(
        'resid_pairs',
        help="Path to whitespace delimited file that is a list of residues to label. Pass in "
             "pairs of residues with the same numbering as in the topology file."
             "Pass multiple lines to model multiple residue pairs")

    # Model Dyes PARAMETERS
    model_parameter_args = model_dyes_parser.add_argument_group("Parameters")
    model_parameter_args.add_argument(
        '--n_procs', required=False, type=int, default=1,
        help="Number of cores to use for parallel processing"
             "Generally parallel over number of frames in supplied trajectory/MSM state")
    model_parameter_args.add_argument(
        '--FRETdye1', required=False,
        default=os.path.dirname(inspect.getfile(ra)) + '/../data/dyes/point-clouds/AF488.pdb',
        help="Path to point cloud of FRET dye pair 2")
    model_parameter_args.add_argument(
        '--FRETdye2', required=False,
        default=os.path.dirname(inspect.getfile(ra)) + '/../data/dyes/point-clouds/AF594.pdb',
        help="Path to point cloud of FRET dye pair 2")
    model_parameter_args.add_argument(
        '--output_dir', required=False, action=readable_dir, default='./',
        help="The location to write the FRET dye distributions.")


    ###########################
    ### Calc_FRET subparser ###
    ###########################
    calc_fret_parser = subparsers.add_parser('calc_FRET',
                                             help='calculate FRET E from MSM centers'
                                                  'using modeled dye distance distribution')

    # Calc FRET INPUTS
    fret_input_args = calc_fret_parser.add_argument_group("Input Settings")
    fret_input_args.add_argument(
        'eq_probs',
        help="equilibrium probabilities from the MSM. "
             "Should be of file type .npy")
    fret_input_args.add_argument(
        't_probs',
        help="transition probabilities from the MSM. "
             "Should be of file type .npy")
    fret_input_args.add_argument(
        'lagtime', type=float,
        help="lag time used to construct the MSM (in ns) "
             "Should be type float")
    fret_input_args.add_argument(
        'FRET_dye_dists', action=readable_dir,
        help="Path to FRET dye distributions (output of model_dyes)")
    fret_input_args.add_argument(
        'resid_pairs',
        help="Path to whitespace delimited file that is a list of residues to label. Pass in "
             "pairs of residues with the same numbering as in the topology file."
             "Pass multiple lines to model multiple residue pairs")

    # Calc FRET PARAMETERS
    fret_parameters = calc_fret_parser.add_argument_group("Parameters")
    fret_parameters.add_argument(
        '--n_procs', required=False, type=int, default=1,
        help="Number of cores to use for parallel processing. "
             "Generally parallel over number of frames in supplied trajectory/MSM state")
    fret_parameters.add_argument(
        '--photon_times',
        default=os.path.dirname(inspect.getfile(ra)) + '/../data/dyes/interphoton_times.npy',
        help="File containing inter photon times. Each list is an individual photon burst "
             "with photon wait times (in us) for each burst. Size (n_bursts, nphotons in burst) "
             "Should be of file type .npy")
    fret_parameters.add_argument(
        '--n_chunks', required=False, type=int, default=2,
        help="Enables you to assess intraburst variation. "
             "How many chunks would you like a given burst broken into?")
    fret_parameters.add_argument(
        '--R0', required=False, type=float, default=5.4,
        help="R0 value for FRET dye pair of interest")
    fret_parameters.add_argument(
        '--time_factor', required=False, type=int, default=1,
        help="factor to slow your trajectories by")
    fret_parameters.add_argument(
        '--output_dir', required=False, action=readable_dir, default='./',
        help="The location to write the FRET dye distributions.")
    fret_parameters.add_argument(
        '--save_burst_frames', required=False, default=False,type=bool,choices=[True,False],
        help='Save a npy file of the frames that make up each burst and the efficiency? T/F')


    ##########################
    ### Fit_FRET subparser ###
    ##########################
    fit_fret_parser = subparsers.add_parser('fit_FRET', help='model FRET dyes onto MSM centers')

    # Fit FRET INPUTS
    fit_FRET_input_args = fit_fret_parser.add_argument_group("Input Settings")

    fit_FRET_input_args.add_argument(
        'fit_conf_file',
        help="Whitespace delimited configuration file for Fit_FRET"
             "Col 1: path to experimental histograms, Col 2: path to output of calc_fret"
             "Repeat for each dye pair in residue file")
    fit_FRET_input_args.add_argument(
        'resid_pairs',
        help="Path to whitespace delimited file that is a list of residues to label. Pass in "
             "pairs of residues with the same numbering as in the topology file."
             "Pass multiple lines to model multiple residue pairs")        

    # Fit FRET PARAMETERS
    fit_FRET_parameters = fit_fret_parser.add_argument_group("Parameters")
    fit_FRET_parameters.add_argument(
        '--method', required=False,
        default='2_3_4_moments',
        choices=['4_moments', '2_3_4_moments', 'sum_sq_residuals', 'entropy'],
        help="Method to use to fit to experimental histogram")
    fit_FRET_parameters.add_argument(
        '--Global_fit', required=False,
        default=False,
        choices=['True', 'False'],
        help="Return the minimum for a global fit?"
             "Won't work if you have different times calculated for each dye pair")
    fit_FRET_parameters.add_argument(
        '--output_dir', required=False, action=readable_dir, default='./',
        help="The location to write the residuals.")


    args = parser.parse_args(argv[1:])
    return args


def main(argv=None):
    args = process_command_line(argv)

    print("Your input was:")
    for i, arg in enumerate(argv):
        # Provide helpful output to remind users their input
        print(i, arg)
    print("", flush=True)

        # Make an output directory
    if args.output_dir != './':
        os.makedirs(args.output_dir, exist_ok=True)

    # Process the input
    if args.command == 'model_dyes':
        # Load Centers and dyes
        trj = md.load(args.centers, top=args.topology)
        logger.info(f"Loaded trajectory {args.centers} using topology file {args.topology}")
        dye1 = dyes_from_expt_dist.load_dye(args.FRETdye1)
        dye2 = dyes_from_expt_dist.load_dye(args.FRETdye2)

        resSeq_pairs = np.loadtxt(args.resid_pairs, dtype=int).reshape(-1,2)

        logger.info(f"Calculating dye distance distribution using dyes: {args.FRETdye1}")
        logger.info(f"and {args.FRETdye2}")
        # Calculate the FRET dye distance distributions for each residue pair
        for n in np.arange(len(resSeq_pairs)):
            logger.info(f"Calculating distance distribution for residue pair: {resSeq_pairs[n]}")
            probs, bin_edges = dyes_from_expt_dist.dye_distance_distribution(
                trj, dye1, dye2, resSeq_pairs[n], n_procs=args.n_procs)
            probs_output = f'{args.output_dir}/probs_{resSeq_pairs[n][0]}_{resSeq_pairs[n][1]}.h5'
            bin_edges_output = f'{args.output_dir}/bin_edges_{resSeq_pairs[n][0]}_{resSeq_pairs[n][1]}.h5'
            ra.save(probs_output, probs)
            ra.save(bin_edges_output, bin_edges)
        logger.info(f"Success! FRET dye distance distributions may be found here: {args.output_dir}")

    elif args.command == 'calc_FRET':
        # Load necessary data
        t_probabilities = np.load(args.t_probs)
        logger.info(f"Loaded t_probs from {args.t_probs}")
        populations = np.load(args.eq_probs)
        logger.info(f"Loaded eq_probs from {args.eq_probs}")
        resSeq_pairs = np.loadtxt(args.resid_pairs, dtype=int).reshape(-1,2)

        cumulative_times = np.load(args.photon_times, allow_pickle=True)

        # Convert Photon arrival times into MSM steps.
        MSM_frames = dyes_from_expt_dist.convert_photon_times(cumulative_times, args.lagtime, args.time_factor)

        logger.info(f"Using r0 of {args.R0}")
        logger.info(f"Using time factor of {args.time_factor}")
        # Calculate the FRET efficiencies
        for n in np.arange(resSeq_pairs.shape[0]):
            logger.info(f"Calculating FRET Efficiencies for residues {resSeq_pairs[n]}")

            title = f'{resSeq_pairs[n, 0]}_{resSeq_pairs[n, 1]}'
            probs_file = f"{args.FRET_dye_dists}/probs_{title}.h5"
            bin_edges_file = f"{args.FRET_dye_dists}/bin_edges_{title}.h5"

            logger.info(f"Loading probs file from {probs_file}")
            probs = ra.load(probs_file)
            logger.info(f"Loading bins file from {bin_edges_file}")
            bin_edges = ra.load(bin_edges_file)
            dist_distribution = dyes_from_expt_dist.make_distribution(probs, bin_edges)
            FEs_sampling, trajs = dyes_from_expt_dist.sample_FRET_histograms(
                T=t_probabilities, populations=populations, dist_distribution=dist_distribution,
                MSM_frames=MSM_frames, R0=args.R0, n_procs=args.n_procs, n_photon_std=args.n_chunks)
            np.save(f"{args.output_dir}/FRET_E_{title}_time_factor_{args.time_factor}.npy", FEs_sampling)

            if args.save_burst_frames==True:
                np.save(f'{args.output_dir}/syn-trjs-{title}.npy', trajs)

        logger.info(f"Success! Your FRET data can be found here: {args.output_dir}")

    elif args.command == 'fit_FRET':
        # Process the conf file
        conf_file=np.loadtxt(args.fit_conf_file, dtype=str)
        expt_histogram_paths = conf_file[:, 0]
        predicted_histogram_paths = conf_file[:, 1]

        labelpairs = np.loadtxt(args.resid_pairs, dtype=int).reshape(-1,2)

        # Initialize a storage array
        difference_array = []

        # Iteratively calculate the difference between expt and prediction.
        for i, label_pair in enumerate(labelpairs):
            print(f'Calculating differences for {label_pair} using {args.method}.')

            # Find all predicted histograms
            #works regardless of label pair ordering so long as user is consistent in labeling pattern.
            FRET_histos = sorted(glob.glob(f'{predicted_histogram_paths[i]}/*{label_pair[0]}*{label_pair[1]}*.npy'))
            if len(FRET_histos) == 0:
                FRET_histos = sorted(glob.glob(f'{predicted_histogram_paths[i]}/*{label_pair[1]}*{label_pair[0]}*.npy'))

            # Split the filename to find the time_scale. Works if timescale is the last item before the file extension
            try:
                intermediate_timescales = [re.split("[. _]", FRET_histos[i]) for i in range(len(FRET_histos))]
                time_scales = [int(file[-2]) for file in intermediate_timescales]
            except ValueError:
                print(f"Tried to find timescales for {label_pair} using last value before file extension,")
                print("at least one of the files read doesn't follow this pattern")
                print(f"Read files were:")
                for file in FRET_histos:
                    print(file)

            # Load the predicted FRET histograms
            predicted_FRET_histos = np.array([np.load(f"{FRET_histos[n]}")
                                              for n in range(len(time_scales))], dtype='O')

            expt_counts = np.loadtxt(f"{expt_histogram_paths[i]}")
            
            if args.method == 'sum_sq_residuals':
                # Can directly calculate this using histogrammed experimental
                # Histogram the predicted FRET efficiencies according to experimental bins
                expt_probs = expt_counts[:, 1] / np.sum(expt_counts[:, 1])
                predicted_histos = dyes_from_expt_dist.histogram_to_match_expt(predicted_FRET_histos[:, :, 0], expt_counts)
                difference_array.append(dyes_from_expt_dist.Sum_sq_resid(expt_probs, predicted_histos))
            elif args.method == 'entropy':
                # Can directly calculate this using histogrammed experimental
                # Histogram the predicted FRET efficiencies according to experimental bins
                expt_probs = expt_counts[:, 1] / np.sum(expt_counts[:, 1])
                predicted_histos = dyes_from_expt_dist.histogram_to_match_expt(predicted_FRET_histos[:, :, 0], expt_counts)
                ent = [entropy(predicted_histos[i], expt_probs) for i in range(len(predicted_histos))]
                difference_array.append(ent)
            elif args.method == '4_moments':
                #Easiest to calculate this using raw data. Regenerate experimental data
                expt_probs = dyes_from_expt_dist.remake_data_from_hist(expt_counts)
                expt_moments = dyes_from_expt_dist.calc_4_moments(expt_probs)
                pred_moments = dyes_from_expt_dist.calc_4_moments(predicted_FRET_histos[:,0])
                diff = dyes_from_expt_dist.normalize_array((expt_moments - pred_moments) ** 2)
                difference_array.append(np.sum(diff, axis=0))
            elif args.method == '2_3_4_moments':
                #Easiest to calculate this using raw data. Regenerate experimental data
                expt_probs = dyes_from_expt_dist.remake_data_from_hist(expt_counts)
                expt_moments = dyes_from_expt_dist.calc_2_3_4_moments(expt_probs)
                pred_moments = dyes_from_expt_dist.calc_2_3_4_moments(predicted_FRET_histos[:,0])
                diff = dyes_from_expt_dist.normalize_array((expt_moments - pred_moments) ** 2)
                difference_array.append(np.sum(diff, axis=0))
            print(
                f"Minimum difference between experiment and prediction for {label_pair}"
                f" is at time factor: {time_scales[np.argmin(difference_array[i])]}.")
            output_array = np.vstack((np.array(time_scales,dtype='O'), difference_array[i])).T
            np.save(f'{args.output_dir}/{label_pair}_{args.method}.npy', output_array)
            print("")
        if args.Global_fit == 'True':
            # Calculate Global Minimums
            print("----Global Minimization----")
            difference_array = np.array(difference_array)
            abs_diff = np.sum(difference_array, axis=0)
            normd_diff = np.sum(dyes_from_expt_dist.normalize_array(difference_array), axis=0)
            print(
                f"Minimum across all dye pairs, normalizing dye-pair differences"
                f" is at time factor: {time_scales[np.argmin(normd_diff)]}.")
            print(
                f"Minimum across all dye pairs, no normalizing across dye-pairs"
                f" is at time factor: {time_scales[np.argmin(abs_diff)]}.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
