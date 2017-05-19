import os
import tempfile
import hashlib
import shutil

from datetime import datetime

from mdtraj.testing import get_fn

from nose.tools import assert_equal

import numpy as np
from numpy.testing import assert_array_equal

from . import rmsd_cluster
from .. import cards
from ..util import array as ra


def runhelper(args, expected_size):

    td = tempfile.mkdtemp(dir=os.getcwd())
    tf = hashlib.md5(str(datetime.now().timestamp())
                     .encode('utf-8')).hexdigest()[0:8]

    try:
        rmsd_cluster.main([
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

        assigns = ra.load(assignfile)
        if type(assigns) is ra.RaggedArray:
            assert_equal(len(assigns), expected_size[0])
            assert_equal(assigns._data.dtype, np.int)
            assert_array_equal(assigns.lengths, expected_size[1])
        else:
            assert_equal(assigns.shape, expected_size)
            assert_equal(assigns.dtype, np.int)

        distfile = os.path.join(
            td, '-'.join(file_tag + ['distances.h5']))
        assert os.path.isfile(distfile), \
            "Couldn't find %s. Dir contained: %s" % (
            distfile, os.listdir(os.path.dirname(distfile)))

        ctrsfile = os.path.join(
            td, '-'.join(file_tag + ['centers.pkl']))
        assert os.path.isfile(ctrsfile), \
            "Couldn't find %s. Dir contained: %s" % (
            ctrsfile, os.listdir(os.path.dirname(ctrsfile)))

    finally:
        shutil.rmtree(td)
        pass


def test_rmsd_cluster_basic():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--atoms', '(name N or name C or name CA or name H or name O)',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_selection():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--atoms', '(name N or name C or name CA)',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_subsample():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--subsample', '4',
        '--atoms', '(name N or name C or name CA or name H or name O)',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_multiprocess():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--processes', '4',
        '--atoms', '(name N or name C or name CA or name H or name O)',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_partition():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--atoms', '(name N or name C or name CA or name H or name O)',
        '--algorithm', 'khybrid',
        '--partition', '4'],
        expected_size=expected_size)


def test_rmsd_cluster_partition_and_subsample():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--atoms', '(name N or name C or name CA or name H or name O)',
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid',
        '--processes', '4',
        '--subsample', '4',
        '--partition', '4'],
        expected_size=expected_size)


def test_rmsd_cluster_multitop():

    expected_size = (3, (501, 501, 5001))

    # trj is length 5001
    xtc2 = os.path.join(cards.__path__[0], 'test_data', 'trj0.xtc')
    top2 = os.path.join(cards.__path__[0], 'test_data', 'PROT_only.pdb')

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--trajectories', xtc2,
        '--topology', get_fn('native.pdb'),
        '--topology', top2,
        '--atoms', '(name N or name C or name CA or name H or name O) '
                   'and (residue 2)',
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_multitop_partition():

    expected_size = (3, (501, 501, 5001))

    xtc2 = os.path.join(cards.__path__[0], 'test_data', 'trj0.xtc')
    top2 = os.path.join(cards.__path__[0], 'test_data', 'PROT_only.pdb')

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--trajectories', xtc2,
        '--topology', get_fn('native.pdb'),
        '--topology', top2,
        '--atoms', '(name N or name C or name CA or name H or name O) '
                   'and (residue 2)',
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid',
        '--partition', '4',
        '--subsample', '4'],
        expected_size=expected_size)


def test_rmsd_cluster_multitop_multiselection():

    expected_size = (3, (501, 501, 5001))

    xtc2 = os.path.join(cards.__path__[0], 'test_data', 'trj0.xtc')
    top2 = os.path.join(cards.__path__[0], 'test_data', 'PROT_only.pdb')

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--atoms', '(name N or name O) and (residue 2)',
        '--trajectories', xtc2,
        '--topology', top2,
        '--atoms', '(name CA) and (residue 3 or residue 4)',
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid',
        '--subsample', '4'],
        expected_size=expected_size)

    # reverse the order. This will catch some cases where the first
    # selection works on both.
    runhelper([
        '--trajectories', xtc2,
        '--topology', top2,
        '--atoms', '(name CA) and (residue 3 or residue 4)',
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--atoms', '(name N or name O) and (residue 2)',
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid',
        '--subsample', '4'],
        expected_size=(expected_size[0], expected_size[1][::-1]))
