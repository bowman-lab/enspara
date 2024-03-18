"""The smFRET app allows you to convert your MSM into a single-molecule
FRET histogram based on the residue pairs of interest. Users specify
protein MSM structure centers and the transition probabilities between each center.

Users also specify dye MSMs to be used to map onto the protein. Dye lifetimes
will be simulated for each protein center via a Monte Carlo approach and the
resulting lifetimes will be used in a protein Monte Carlo to return the average FRET
efficiency per photon burst as well as the lifetimes of donor and acceptor photons
during that burst. 

Dyes may be mapped to any amino acid in the protein. As single-molecule FRET is 
highly dependent on true-to-experiment simulation timescales, you can also rescale the 
speed of your trajectories within an MSM. This code also enables adaptation to specific
smFRET experimental setups as users provide their own experimental FRET bursts. 
"""
# Author: Justin J Miller <justinjm@seas.upenn.edu>


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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_command_line(argv):
    parser = argparse.ArgumentParser(
        prog='smFRET',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert an MSM and a series of FRET dye residue pairs into"
                    "the predicted FRET efficiency for each pair and fit to expt time"
                    "Multi-step process- calc lifetimes, calc FRET efficiency, fit time factor")

    subparsers = parser.add_subparsers(title='commands', dest='command',
                                       description='valid subcommands',
                                       help='Call a subparser!')

    ############################
    ### Model_dyes subparser ###
    ############################
    calc_lifetimes_parser = subparsers.add_parser('calc_lifetimes', help='model FRET \
    	dyes onto MSM centers and calculate their lifetimes')

    # Model dyes INPUTS
    calc_lts_input_args = model_dyes_parser.add_argument_group("Input Settings")
    calc_lts_input_args.add_argument(
        'donor_centers',
        help="Path to cluster centers from the MSM"
             "should be of type .xtc.")
    calc_lts_input_args.add_argument(
        'donor_top',
        help="topology file for supplied trajectory")
    calc_lts_input_args.add_argument(
        'acceptor_centers',
        help="Path to cluster centers from the MSM"
             "should be of type .xtc.")
    calc_lts_input_args.add_argument(
        'acceptor_top',
        help="topology file for supplied trajectory")
    calc_lts_input_args.add_argument(
        'dye_lagtime', type=float
        help="Lagtime for dye MSMs")
    calc_lts_input_args.add_argument(
    	'Prot_Centers',
    	help='Protein MSM cluster centers, should be readable by mdtraj')
    calc_lts_input_args.add_argument(
    	'Prot_top',
    	help='Protein topology file to read protein centers')

    ### Better way to do this?
    calc_lts_input_args.add_argument(
        'dyenames',
        help="Name of dyes to label with")
    calc_lts_input_args.add_argument(
        'resid_pairs',
        help="Path to whitespace delimited file that is a list of residues to label. Pass in "
             "pairs of residues with the same numbering as in the topology file."
             "Pass multiple lines to model multiple residue pairs")

    # Optional PARAMETERS
    model_parameter_args = model_dyes_parser.add_argument_group("Parameters")
    model_parameter_args.add_argument(
        '--n_procs', required=False, type=int, default=1,
        help="Number of cores to use for parallel processing"
             "Generally parallel over number of frames in supplied trajectory/MSM state")
    model_parameter_args.add_argument(
        '--n_samples', required=False, type=int,
        default=1000,
        help="Number of times to run dye_lifetime calculations (per center)")
    model_parameter_args.add_argument(
        '--save_dtrj', required=False, default=False, type=bool,
        help="Save dye trajectories and sampled states? Saves per protein center.")
    model_parameter_args.add_argument(
        '--save_dmsm', required=False, default=False, type=bool,
        help="Save dye MSMs with steric clash states dropped out? Saves per protein center.")    
    model_parameter_args.add_argument(
        '--output_dir', required=False, action=readable_dir, default='./',
        help="Location to write output to.")
#### Made it to here!



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
        'photon_times',
        help="File containing inter photon times. Each list is an individual photon burst "
             "with photon wait times (in us) for each burst. Size (n_bursts, nphotons in burst) "
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
        '--n_chunks', required=False, type=int, default=2,
        help="Enables you to assess intraburst variation. "
             "How many chunks would you like a given burst broken into?")
    fret_parameters.add_argument(
        '--R0', required=False, type=float, default=5.4,
        help="R0 value for FRET dye pair of interest")
    fret_parameters.add_argument(
        '--slowing_factor', required=False, type=int, default=1,
        help="factor to slow your trajectories by")
    fret_parameters.add_argument(
        '--output_dir', required=False, action=readable_dir, default='./',
        help="The location to write the FRET dye distributions.")
    fret_parameters.add_argument(
        '--save_prot_trj', required=False, default=False, type=bool,
        help="Save center indicies of protein states visited during each burst?")


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
        try:
            os.system(f'mkdir {args.output_dir}')
        except:
            pass

    # Process the input
    if args.command == 'model_dyes':


    elif args.command == 'calc_FRET':


    elif args.command == 'fit_FRET':



if __name__ == "__main__":
    sys.exit(main(sys.argv))