import numpy as np
import mdtraj as md

from nose.tools import assert_is, assert_is_not

from numpy.testing import assert_array_equal, assert_allclose

from enspara.cluster import util
from enspara.util import array as ra

from ..cluster import save_states
from .util import get_fn


def test_ClusterResult_partition_np():
    list_lens = [20, 20, 20]

    concat_assigs = [0]*20 + [1]*20 + [2]*20
    concat_dists = [0.2]*20 + [0.3]*20 + [0.4]*20
    concat_ctr_inds = [3, 23, 43]

    concat_rslt = util.ClusterResult(
        assignments=concat_assigs,
        distances=concat_dists,
        center_indices=concat_ctr_inds,
        centers=None)

    rslt = concat_rslt.partition(list_lens)

    # ensuring it isn't a ragged array allows list, ndarray or maskedarray
    assert_is_not(type(rslt.assignments), ra.RaggedArray)
    assert_array_equal(rslt.assignments[0], [0]*20)
    assert_array_equal(rslt.assignments[1], [1]*20)
    assert_array_equal(rslt.assignments[2], [2]*20)

    assert_is_not(type(rslt.distances), ra.RaggedArray)
    assert_array_equal(rslt.distances[0], [0.2]*20)
    assert_array_equal(rslt.distances[1], [0.3]*20)
    assert_array_equal(rslt.distances[2], [0.4]*20)

    assert_array_equal(rslt.center_indices, [(0, 3), (1, 3), (2, 3)])


def test_ClusterResult_partition_ra():
    list_lens = [10, 20, 100]

    concat_assigs = [0]*10 + [1]*20 + [2]*100
    concat_dists = [0.2]*10 + [0.3]*20 + [0.4]*100
    concat_ctr_inds = [3, 23, 103]

    concat_rslt = util.ClusterResult(
        assignments=concat_assigs,
        distances=concat_dists,
        center_indices=concat_ctr_inds,
        centers=None)

    rslt = concat_rslt.partition(list_lens)

    assert_is(type(rslt.assignments), ra.RaggedArray)
    assert_array_equal(rslt.assignments[0], [0]*10)
    assert_array_equal(rslt.assignments[1], [1]*20)
    assert_array_equal(rslt.assignments[2], [2]*100)

    assert_is(type(rslt.distances), ra.RaggedArray)
    assert_array_equal(rslt.distances[0], [0.2]*10)
    assert_array_equal(rslt.distances[1], [0.3]*20)
    assert_array_equal(rslt.distances[2], [0.4]*100)

    assert_array_equal(rslt.center_indices, [(0, 3), (1, 13), (2, 73)])


def test_unique_state_extraction():
    '''
    Check to makes sure we get the unique states from the trajectory
    correctly
    '''

    states = [0, 1, 2, 3, 4]
    assignments = np.random.choice(states, (100000))

    assert all(save_states.unique_states(assignments) == states)

    states = [-1, 0, 1, 2, 3, 4]
    assignments = np.random.choice(states, (100000))

    assert all(save_states.unique_states(assignments) == states[1:])


def test_assign_to_nearest_center_few_centers():

    # assign_to_nearest_center takes two code paths, one for
    # n_centers > n_frames and one for n_frames > n_centers. This tests
    # the latter.
    trj = md.load(get_fn('frame0.xtc'), top=get_fn('native.pdb'))
    center_frames = [0, int(len(trj)/3), int(len(trj)/2)]

    assigns, distances = util.assign_to_nearest_center(
        trj, trj[center_frames], md.rmsd)

    alldists = np.zeros((len(center_frames), len(trj)))
    for i, center_frame in enumerate(trj[center_frames]):
        alldists[i] = md.rmsd(trj, center_frame)

    assert_allclose(np.min(alldists, axis=0), distances, atol=1e-3)
    assert_array_equal(np.argmin(alldists, axis=0), assigns)


def test_assign_to_nearest_center_many_centers():

    # assign_to_nearest_center takes two code paths, one for
    # n_centers > n_frames and one for n_frames > n_centers. This tests
    # the former.
    trj = md.load(get_fn('frame0.xtc'), top=get_fn('native.pdb'))[::10]
    center_frames = list(range(len(trj))) + list(range(len(trj) // 2))

    assigns, distances = util.assign_to_nearest_center(
        trj, trj[center_frames], md.rmsd)

    alldists = np.zeros((len(center_frames), len(trj)))
    for i, center_frame in enumerate(trj[center_frames]):
        alldists[i] = md.rmsd(trj, center_frame)

    assert_allclose(np.min(alldists, axis=0), distances, atol=1e-3)
    assert_array_equal(np.argmin(alldists, axis=0), assigns)


def test_find_cluster_centers_ndarray():

    d = np.array([0.2, 0.1, 0.1, 0.2])
    a = np.array([1, 1, 7, 7])

    ctrs = util.find_cluster_centers(assignments=a, distances=d)

    assert_array_equal(ctrs, [1, 2])
