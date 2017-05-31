import sys
import argparse
import os
import logging
import itertools
import pickle
import json

from functools import partial
from multiprocessing import cpu_count

import numpy as np
import mdtraj as md

from enspara.cluster import KHybrid
from enspara.util import array as ra
from enspara.cluster.util import load_frames
from enspara.util import load_as_concatenated
from enspara import exception

from enspara.apps.reassign import reassign

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_command_line(argv):
    '''Parse the command line and do a first-pass on processing them into a
    format appropriate for the rest of the script.'''

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cluster a set (or several sets) of trajectories "
                    "into a single state space based upon RMSD.")

    parser.add_argument(
        '--trajectories', required=True, nargs="+", action='append',
        help="List of paths to aligned trajectory files to cluster. "
             "All file types that MDTraj supports are supported here.")
    parser.add_argument(
        '--topology', required=True, action='append', dest='topologies',
        help="The topology file for the trajectories. This flag must be"
             " specified once for each instance of the --trajectories "
             "flag. The first --topology flag is taken to be the "
             "topology to use for the first instance of the "
             "--trajectories flag, and so forth.")
    parser.add_argument(
        '--algorithm', required=True, choices=["khybrid"],
        help="The clustering algorithm to use.")
    parser.add_argument(
        '--atoms', action="append", required=True,
        help="The atoms from the trajectories (using MDTraj "
             "atom-selection syntax) to cluster based upon. Specify "
             "once to apply this selection to every set of "
             "trajectories specified by the --trajectories flag, or "
             "once for each different topology (i.e. the number of "
             "times --trajectories and --topology was specified.)")
    parser.add_argument(
        '--rmsd-cutoff', required=True, type=float,
        help="The RMSD cutoff to determine cluster size. Units: nm.")
    parser.add_argument(
        '--processes', default=cpu_count(), type=int,
        help="Number processes to use for loading and clustering.")
    parser.add_argument(
        '--partitions', default=None, type=int,
        help="Use this number of pivots when delegating to md.rmsd. "
             "This can avoid md.rmsd's large-dataset segfault.")
    parser.add_argument(
        '--subsample', default=None, type=int,
        help="Take only every nth frame when loading trajectories. "
             "1 implies no subsampling.")
    parser.add_argument(
        '--output-path', default='',
        help="The output path for results (distances, assignments, centers).")
    parser.add_argument(
        '--output-tag', default='',
        help="An optional extra string prepended to output filenames (useful"
             "for giving this choice of parameters a name to separate it from"
             "other clusterings or proteins.")

    args = parser.parse_args(argv[1:])

    if len(args.atoms) == 1:
        args.atoms = args.atoms * len(args.trajectories)
    elif len(args.atoms) != len(args.trajectories):
        raise exception.ImproperlyConfigured(
            "Flag --atoms must be provided either once (selection is "
            "applied to all trajectories) or the same number of times "
            "--trajectories is supplied.")

    if len(args.topologies) != len(args.trajectories):
        raise exception.ImproperlyConfigured(
            "The number of --topology and --trajectory flags must agree.")

    return args


def rmsd_hack(trj, ref, partitions=None, **kwargs):

    if partitions is None:
        return md.rmsd(trj, ref)

    pivots = np.linspace(0, len(trj), num=partitions+1, dtype='int')

    rmsds = np.zeros(len(trj))

    for i in range(len(pivots)-1):
        s = slice(pivots[i], pivots[i+1])
        rmsds[s] = md.rmsd(trj[s], ref, **kwargs)

    return rmsds


def filenames(args):

    tag_params = [
        args.output_tag,
        args.algorithm,
        str(args.rmsd_cutoff)]

    if args.subsample:
        tag_params.append(str(args.subsample)+'subsample')

    path_stub = os.path.join(args.output_path, '-'.join(tag_params))

    return {
        'distances': path_stub+'-distances.h5',
        'centers': path_stub+'-centers.pkl',
        'assignments': path_stub+'-assignments.h5',
    }


def load(topologies, trajectories, selections, stride, processes):

    for top, selection in zip(topologies, selections):
        sentinel_trj = md.load(top)
        try:
            # noop, but causes fast-fail w/bad args.atoms
            sentinel_trj.top.select(selection)
        except:
            raise exception.ImproperlyConfigured((
                "The provided selection '{s}' didn't match the topology "
                "file, {t}").format(s=selection, t=top))

    flat_trjs = []
    configs = []
    for topfile, trjset, selection in zip(topologies, trajectories,
                                          selections):
        top = md.load(topfile).top
        for trj in trjset:
            flat_trjs.append(trj)
            configs.append({
                'top': top,
                'stride': stride,
                'atom_indices': top.select(selection),
                'selection': selection,
                })

    assert all([len(c['atom_indices']) == len(configs[0]['atom_indices'])
                for c in configs]), \
        "Number of atoms across different input topologies differed: %s" % \
        [(t, c['atom_indices'], c['selection'])
         for t, c in zip(topologies, configs)]

    logger.info(
        "Loading %s trajectories with %s atoms using %s processes"
        "(subsampling %s)",
        len(flat_trjs), len(top.select(selection)), processes, stride)
    assert len(top.select(selection)) > 0, "No atoms selected for clustering"

    lengths, xyz = load_as_concatenated(
        flat_trjs, args=configs, processes=processes)

    logger.info(
        "Loaded %s frames.", len(xyz))

    return lengths, xyz, top.subset(top.select(selection))


def load_asymm_frames(result, trajectories, topology, subsample):

    frames = []
    begin_index = 0
    for topfile, trjset in zip(topology, trajectories):
        end_index = begin_index + len(trjset)
        subframes = load_frames(
            list(itertools.chain(*trajectories)),
            [c for c in result.center_indices
             if begin_index <= c[0] < end_index],
            top=md.load(topfile).top,
            stride=subsample)

        frames.extend(subframes)
        begin_index += len(trjset)

    return frames


def position_of_first_difference(paths):
    for i, chars in enumerate(zip(*paths)):
        if not all(chars[0] == char_i for char_i in chars):
            break

    return i


def main(argv=None):
    '''Run the driver script for this module. This code only runs if we're
    being run as a script. Otherwise, it's silent and just exposes methods.'''
    args = process_command_line(argv)

    i = position_of_first_difference(args.topologies)

    targets = {topf[i:]: "%s xtcs" % len(trjfs) for topf, trjfs
               in zip(args.topologies, args.trajectories)}
    logger.info("Beginning RMSD Clusutering app. Operating on targets:\n%s",
                json.dumps(targets, indent=4))

    lengths, xyz, select_top = load(
        args.topologies, args.trajectories, selections=args.atoms,
        stride=args.subsample, processes=args.processes)

    logger.info(
        "Loading finished. Clustering using atoms %s matching '%s'.",
        xyz.shape[1], args.atoms)

    clustering = KHybrid(
        metric=partial(rmsd_hack, partitions=args.partitions),
        cluster_radius=args.rmsd_cutoff)

    # md.rmsd requires an md.Trajectory object, so wrap `xyz` in
    # the topology.
    clustering.fit(md.Trajectory(xyz=xyz, topology=select_top))

    logger.info(
        "Clustered %s frames into %s clusters in %s seconds.",
        sum(lengths), len(clustering.centers_), clustering.runtime_)

    result = clustering.result_.partition(lengths)

    outdir = os.path.dirname(filenames(args)['centers'])
    logger.info("Saving cluster centers at %s", outdir)

    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    with open(filenames(args)['centers'], 'wb') as f:
        centers = load_asymm_frames(result, args.trajectories, args.topologies,
                                    args.subsample)
        pickle.dump(centers, f)

    if args.subsample:
        # overwrite temporary output with actual results
        assig, dist = reassign(
            args.topologies, args.trajectories, args.atoms,
            centers=result.centers)

        ra.save(filenames(args)['distances'], dist)
        ra.save(filenames(args)['assignments'], assig)
    else:
        ra.save(filenames(args)['distances'], result.distances)
        ra.save(filenames(args)['assignments'], result.assignments)

    logger.info("Success! Data can be found in %s.",
                os.path.dirname(filenames(args)['distances']))

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
