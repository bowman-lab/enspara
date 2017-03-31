import os
import tempfile
import hashlib
import shutil

from datetime import datetime

from nose.tools import assert_equal, assert_is
from mdtraj.testing import get_fn

from rmsd_cluster import main as rmsd_cluster


def runhelper(args):

    td = tempfile.mkdtemp(dir=os.getcwd())
    tf = hashlib.md5(str(datetime.now().timestamp()) \
            .encode('utf-8')).hexdigest()[0:8]

    try:
        rmsd_cluster([
            '',
            '--output-path', td,
            '--output-tag', tf] + args)

        assignfile = os.path.join(
            td, '-'.join([tf, 'khybrid', '0.1', 'assignments.h5']))
        assert os.path.isfile(assignfile), "Couldn't find %s" % assignfile

        distfile = os.path.join(
            td, '-'.join([tf, 'khybrid', '0.1', 'distances.h5']))
        assert os.path.isfile(distfile), "Couldn't find %s" % distfile

        ctrsfile = os.path.join(
            td, '-'.join([tf, 'khybrid', '0.1', 'centers.h5']))
        assert os.path.isfile(ctrsfile), "Couldn't find %s" % ctrsfile

    finally:
        shutil.rmtree(td)
        pass


def test_rmsd_cluster_basic():

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
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

