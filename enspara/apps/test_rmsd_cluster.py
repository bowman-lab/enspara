import os
import tempfile
import hashlib
import shutil

from datetime import datetime

from mdtraj.testing import get_fn

from .rmsd_cluster import main as rmsd_cluster
from .. import cards


def runhelper(args):

    td = tempfile.mkdtemp(dir=os.getcwd())
    tf = hashlib.md5(str(datetime.now().timestamp())
                     .encode('utf-8')).hexdigest()[0:8]

    try:
        rmsd_cluster([
            '',  # req'd because arg[0] is expected to be program name
            '--output-path', td,
            '--output-tag', tf] + args)

        file_tag = [tf, 'khybrid', '0.1']

        # append the subsample to the expected output tags when relevant
        if '--subsample' in args:
            file_tag.append(
                args[args.index('--subsample')+1]+'subsample')

        assignfile = os.path.join(
            td, '-'.join(file_tag + ['assignments.h5']))
        assert os.path.isfile(assignfile), \
            "Couldn't find %s. Dir contained: %s" % (
            assignfile, os.listdir(os.path.dirname(assignfile)))

        distfile = os.path.join(
            td, '-'.join(file_tag + ['distances.h5']))
        assert os.path.isfile(distfile), \
            "Couldn't find %s. Dir contained: %s" % (
            distfile, os.listdir(os.path.dirname(distfile)))

        ctrsfile = os.path.join(
            td, '-'.join(file_tag + ['centers.h5']))
        assert os.path.isfile(ctrsfile), \
            "Couldn't find %s. Dir contained: %s" % (
            ctrsfile, os.listdir(os.path.dirname(ctrsfile)))

    finally:
        shutil.rmtree(td)
        pass


def test_rmsd_cluster_basic():

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid'])


def test_rmsd_cluster_selection():

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--atoms', '(name N or name C or name CA)',
        '--algorithm', 'khybrid'])


def test_rmsd_cluster_subsample():

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--subsample', '4',
        '--algorithm', 'khybrid'])


def test_rmsd_cluster_multiprocess():

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--processes', '4',
        '--algorithm', 'khybrid'])


def test_rmsd_cluster_partition():

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid',
        '--partition', '4'])


def test_rmsd_cluster_partition_and_subsample():

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid',
        '--processes', '4',
        '--subsample', '4',
        '--partition', '4'])


def test_rmsd_cluster_multitop():

    xtc2 = os.path.join(cards.__path__[0], 'test_data', 'trj0.xtc')
    top2 = os.path.join(cards.__path__[0], 'test_data', 'PROT_only.pdb')

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--trajectories', xtc2,
        '--topology', get_fn('native.pdb'),
        '--topology', top2,
        '--atoms', '(name N or name C or name CA or name H or name O)'
                   'and (residue 2)',
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid'])


def test_rmsd_cluster_multitop_partition():

    xtc2 = os.path.join(cards.__path__[0], 'test_data', 'trj0.xtc')
    top2 = os.path.join(cards.__path__[0], 'test_data', 'PROT_only.pdb')

    print(get_fn('native.pdb'))

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--trajectories', xtc2,
        '--topology', get_fn('native.pdb'),
        '--topology', top2,
        '--atoms', '(name N or name C or name CA or name H or name O)'
                   'and (residue 2)',
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid',
        '--partition', '4',
        '--subsample', '4'])