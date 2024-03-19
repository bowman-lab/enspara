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
import mdtraj as md
import numpy as np
import enspara
from functools import partial
from multiprocessing import get_context
from enspara.geometry import dyes_from_expt_dist as dyefs
from enspara.geometry import dye_lifetimes
from enspara.apps.util import readable_dir

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

    ################################
    ### calc lifetimes subparser ###
    ################################
    calc_lifetimes_parser = subparsers.add_parser('calc_lifetimes', help='model FRET \
        dyes onto MSM centers and calculate their lifetimes')

    # Model dyes INPUTS
    calc_lts_input_args = calc_lifetimes_parser.add_argument_group("Input Settings (Required)")
    calc_lts_input_args.add_argument(
        '--donor_name', required=True,
        help="Name of the donor dye. Should be in the enspara dye library.")
    calc_lts_input_args.add_argument(
        '--donor_centers', required=True,
        help="Path to cluster centers from the MSM"
             "should be of type .xtc.")
    calc_lts_input_args.add_argument(
        '--donor_top', required=True,
        help="topology file for supplied trajectory")
    calc_lts_input_args.add_argument(
        '--donor_tcounts', required=True,
        help='t_counts for the donor dye MSM.')
    calc_lts_input_args.add_argument(
        '--acceptor_name', required=True,
        help='Name of the acceptor dye. Should be in the enspara dye library.')
    calc_lts_input_args.add_argument(
        '--acceptor_centers', required=True,
        help="Path to cluster centers from the MSM"
             "should be of type .xtc.")
    calc_lts_input_args.add_argument(
        '--acceptor_top', required=True,
        help="topology file for supplied trajectory")
    calc_lts_input_args.add_argument(
        '--acceptor_tcounts', required=True,
        help='t_counts for the acceptor dye MSM')
    calc_lts_input_args.add_argument(
        '--dye_lagtime', type=float, required=True,
        help="Lagtime for dye MSMs, in ns."
        "Enspara dye MSMs were built with a lagtime of 0.002 ns.")
    calc_lts_input_args.add_argument(
        '--prot_top', required=True,
        help='Protein topology file to read protein centers')
    calc_lts_input_args.add_argument(
        '--resid_pairs', required=True,
        help="Path to whitespace delimited file that is a list of residues to label. Pass in "
             "pairs of residues with the same numbering as in the topology file."
             "Pass multiple lines to model multiple residue pairs")

    # Optional PARAMETERS
    calc_lts_param_args = calc_lifetimes_parser.add_argument_group("Parameters (Optional)")
    calc_lts_param_args.add_argument(
        '--prot_centers', required=False,
        help="Path to protein MSM cluster centers."
        "Should be trajectory file readable by mdtraj."
        "If not provided, will just label the protein topology file."
        "Running burst with a single protein center is not supported though, since there are no"
        "conformations to average over. Calculate FRET directly from the lifetime outcomes.")
    calc_lts_param_args.add_argument(
        '--n_procs', required=False, type=int, default=1,
        help="Number of cores to use for parallel processing"
             "Generally parallel over number of frames in supplied trajectory/MSM state")
    calc_lts_param_args.add_argument(
        '--n_samples', required=False, type=int,
        default=5000,
        help="Number of times to run dye_lifetime calculations (per center)")
    calc_lts_param_args.add_argument(
        '--save_dtrj', required=False, default=False, type=bool,
        help="Save dye trajectories and sampled states? Saves per protein center.")
    calc_lts_param_args.add_argument(
        '--save_dmsm', required=False, default=False, type=bool,
        help="Save dye MSMs with steric clash states dropped out? Saves per protein center.")    
    calc_lts_param_args.add_argument(
        '--output_dir', required=False, action=readable_dir, default='./',
        help="Location to write output to.")


    ###########################
    ### Run Burst subparser ###
    ###########################
    run_burst_parser = subparsers.add_parser('run_burst',
                                             help='calculate FRET E from MSM centers'
                                                  'using modeled dye lifetimes')

    # Calc FRET INPUTS
    burst_input_args = run_burst_parser.add_argument_group("Input Settings (Required)")
    burst_input_args.add_argument(
        '--eq_probs', required=True,
        help="Path to equilibrium probabilities from the protein MSM. "
             "Should be of file type .npy")
    burst_input_args.add_argument(
        '--t_counts', required=True,
        help="Path to transition counts from the protein MSM. "
             "Should be of file type .npy")
    burst_input_args.add_argument(
        '--prot_centers', required=True,
        help="Path to protein MSM cluster centers."
        "Should be trajectory file readable by mdtraj.")
    burst_input_args.add_argument(
        '--prot_top', required=True,
        help="Path to protein topology file.")
    burst_input_args.add_argument(
        '--lifetimes_dir', action=readable_dir,
        help="Path to dye-lifetimes directory / output from calc_lifetimes.")
    burst_input_args.add_argument(
        '--donor_name', type=str,  required=True,
        help="Name of donor dye. Should be a dye in the Enspara dye library.")
    burst_input_args.add_argument(
        '--acceptor_name', type=str, required=True,
        help="Name of acceptor dye. Should be a dye in the Enspara dye library.")
    burst_input_args.add_argument(
        '--lagtime', type=float, required=True,
        help="lag time used to construct the protein MSM (in ns) "
             "Should be type float")
    burst_input_args.add_argument(
        '--FRET_dye_dists', action=readable_dir, required=True,
        help="Path to FRET dye distributions (output of model_dyes)")
    burst_input_args.add_argument(
        '--resid_pairs', required=True,
        help="Path to whitespace delimited text file that is a list of residues to label. Pass in "
             "pairs of residues with the same numbering (resSeq) as in the topology file."
             "Pass multiple lines to model multiple residue pairs. First residue is the donor"
             "and the second residue corresponds to the acceptor.")

    # Calc FRET PARAMETERS
    burst_parameters = run_burst_parser.add_argument_group("Parameters (Optional)")
    burst_parameters.add_argument(
        '--n_procs', required=False, type=int, default=1,
        help="Number of cores to use for parallel processing. "
             "Generally parallel over number of labeled residues.")
    burst_parameters.add_argument(
        '--output_dir', required=False, action=readable_dir, default='./',
        help="The location to write the FRET dye distributions.")
    burst_parameters.add_argument(
        '--photon_times', required=False, 
        default=f'{os.path.dirname(enspara.__file__)}/data/dyes/interphoton_times.npy',
        help="File containing inter photon times. Each list is an individual photon burst "
             "with photon wait times (in us) for each burst. Size (n_bursts, nphotons in burst) "
             "Should be of file type .npy")
    burst_parameters.add_argument(
        '--correction_factor', required=False, type=int, default=10000, 
        nargs="+", action='append',
        help="Time factor by which your MSM is faster than experimental timescale."
        "Pass multiple to rescale MSM to multiple times.")

    args = parser.parse_args(argv[1:])
    return args


def main(argv=None):
    args = process_command_line(argv)

    os.environ['NUMEXPR_MAX_THREADS'] = str(args.n_procs)
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    print("Your input was:")
    for i, arg in enumerate(argv):
        # Provide helpful output to remind users their input
        print(i, arg)
    print("", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    resSeqs = np.loadtxt(args.resid_pairs, dtype=int).reshape(-1,2)

    # Process the input
    if args.command == 'calc_lifetimes':
        #Load in initial stuff
        print('Loading dye MSMs.', flush=True)
        d_centers = md.load(args.donor_centers, top=args.donor_top)
        a_centers = md.load(args.acceptor_centers, top=args.acceptor_top)
        d_tcounts = np.load(args.donor_tcounts, allow_pickle=True)
        a_tcounts = np.load(args.acceptor_tcounts, allow_pickle=True)

        print('Loading protein centers.', flush=True)
        if args.prot_centers == None:
            prot_traj = md.load(args.prot_top)
        else:
            prot_traj = md.load(args.prot_centers, top=args.prot_top)

        for resSeq in resSeqs:
            func = partial(dye_lifetimes.calc_lifetimes, d_centers=d_centers, d_tcounts=d_tcounts,
            a_centers=a_centers, a_tcounts=a_tcounts, resSeqs=resSeq, 
            dyenames=[args.donor_name, args.acceptor_name],
            dye_lagtime=args.dye_lagtime, n_samples=args.n_samples, outdir=args.output_dir, 
            save_dye_trj=args.save_dtrj, save_dye_msm=args.save_dmsm)

            print(f'Starting pool for resSeq {resSeq}.', flush=True)

            procs = min([len(prot_traj), args.n_procs])

            with get_context("spawn").Pool(processes = procs) as pool:

                lifetime_events = pool.map(func, zip(prot_traj, np.arange(len(prot_traj))))
                pool.terminate()

            lifetime_events = np.array(lifetime_events)
            print(f'Saving lifetimes and outcomes here: {args.output_dir}')
            np.save(f'{args.output_dir}/events-{resSeq[0]}-{resSeq[1]}.npy', lifetime_events)


    elif args.command == 'run_burst':

   		#Load in initial files
        prot_traj=md.load(args.prot_top)
        prot_tcounts = np.load(args.t_counts, allow_pickle=True)
        prot_eqs = np.load(args.eq_probs)
        cumulative_times = np.load(args.photon_times)

        #Make output dirs
        os.makedirs(f'{args.output_dir}/MSMs', exist_ok=True)

        #Choose a sensible number of processes to start for pool.
        procs = min([len(resSeqs),args.n_procs])

        print('Remaking dye MSMs to account for protein states with no available dyes.', flush=True)

        #Remake dye MSM for each dye pair
        func = partial(dye_lifetimes.remake_msms, prot_tcounts=prot_tcounts, dye_dir=args.lifetimes_dir,
            dyenames=[args.donor_name, args.acceptor_name],orig_eqs=prot_eqs, outdir = args.output_dir)
        with get_context("spawn").Pool(processes=procs) as pool:
            run = pool.map(func, resSeqs)
            pool.terminate()

        #Run burst MC for each correction factor
        for time_correction in args.correction_factor:
            # Convert Photon arrival times into MSM steps.
            MSM_frames = dyefs.convert_photon_times(cumulative_times, args.lagtime, time_correction)

            func = partial(dye_lifetimes.run_mc, prot_tcounts=prot_tcounts, 
               dyenames=[args.donor_name, args.acceptor_name], 
                dye_dir=args.lifetimes_dir, orig_eqs=prot_eqs, MSM_frames=MSM_frames, 
                outdir=args.output_dir, time_correction=time_correction)

            with get_context("spawn").Pool(processes=procs) as pool:
                run = pool.map(func, resSeqs)
                pool.terminate()

if __name__ == "__main__":
    sys.exit(main(sys.argv))
