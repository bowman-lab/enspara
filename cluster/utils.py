# Authors: Maxwell I. Zimmerman <mizimmer@wustl.edu>,
#          Gregory R. Bowman <gregoryrbowman@gmail.com>,
#          Justin R. Porter <justinrporter@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

from functools import wraps
import sys
import time
from collections import namedtuple

import mdtraj as md
import numpy as np

from ..exception import ImproperlyConfigured, DataInvalid
from ..traj_manipulation import sloopy_concatenate_trjs


Clustering = namedtuple(
    'Clustering', ['center_indices', 'assignments', 'distances'])


def assign_to_nearest_center(traj, cluster_centers, distance_method):
    n_frames = len(traj)
    assignments = np.zeros(n_frames, dtype=int)
    distances = np.empty(n_frames, dtype=float)
    distances.fill(np.inf)
    cluster_center_inds = []

    cluster_num = 0
    for center in cluster_centers:
        dist = distance_method(traj, center)
        inds = (dist < distances)
        distances[inds] = dist[inds]
        assignments[inds] = cluster_num
        new_center_index = np.argmin(dist)
        cluster_center_inds.append(new_center_index)
        cluster_num += 1

    return cluster_center_inds, assignments, distances


def load_frames(indices, filenames, **kwargs):
    '''
    Given a list of trajectory file names (`filenames`) and tuples
    indicating trajectory number and frame number (`center indices`),
    load the given frames into a list of md.Trajectory objects. All
    additional kwargs are passed on to md.load_frame.
    '''

    if 'stride' in kwargs:
        stride = int(kwargs['stride'])

    centers = [md.load_frame(filenames[i], index=j*stride, **kwargs)
               for i, j in indices]
    return centers


def find_cluster_centers(traj_lst, distances):
    '''
    Use the distances matrix to calculate which of the trajectories in
    traj_lst are the center of a cluster. Return a list of indices.
    '''

    if len(traj_lst) != distances.shape[0]:
        raise DataInvalid(
            "Expected len(traj_lst) {} to match distances.shape[0] ({})".
            format(len(traj_lst), distances.shape))

    center_indices = np.argwhere(distances == 0)

    centers = [traj_lst[trj_indx][frame] for (trj_indx, frame)
               in center_indices]

    assert len(centers) == center_indices.shape[0]

    return centers


def _delete_trajs_warning(traj_lst, write_to=sys.stdout):
    '''
    Inspect traj_list and determine if the user should be warned.
    '''
    if hasattr(traj_lst[0], 'n_atoms'):
        atom_count = sum([t.n_atoms*t.n_frames for t in traj_lst])
    else:
        atom_count = sum([reduce(lambda x, y: x*y, t.shape)
                          for t in traj_lst])
    # TODO: this warning is currently set too low. If you're
    # enountering this error needlessly, bump it up!
    if atom_count > 1e8:
        sys.stdout.write(
            "WARNING: There are a lot of atoms/frames (%s "
            "atom-frames) in this system. Consider using "
            "delete_trjs=True to save memory.\n" % atom_count)


def requires_concatenated_trajectories(cluster_algo):
    '''
    This decorator takes a cluster_algorithm and wraps it in boilerplate code
    that needs to precede/follow any clustering algorithm. In particular, it
    concatenates and de-concatenate (repartition) trajectories before and after
    running, and it determines the distance_metric callable. It also filters
    out the cluster centers return value, to prevent silliness. It also offers
    the parameter `delete_trjs` which will delete the trajectories in traj_lst
    as it works to preserve memory.
    '''

    @wraps(cluster_algo)  # @wraps corrects f.__name__ and whatnot
    def cluster_fn(traj_lst, metric='rmsd', delete_trjs=False,
                   output=sys.stdout, *args, **kwargs):

        # cache start time for use at the end
        start = time.clock()

        if not delete_trjs:
            _delete_trajs_warning(traj_lst)

        assert hasattr(output, 'write'), "The parameter 'output' must provide a 'write' method for doing output."
        distance_method = _get_distance_method(metric)


        # concatenate trajectories
        traj_lengths = [len(t) for t in traj_lst]
        concat_start = time.clock()

        if isinstance(traj_lst[0], md.Trajectory) and len(traj_lst) > 0:
            traj = sloopy_concatenate_trjs(traj_lst, delete_trjs=delete_trjs)
        else:
            traj = np.concatenate(traj_lst)

        output.write("Concatenated in %s seconds.\n" %
                     (time.clock() - concat_start))


        # go, cluster, go!
        cluster_center_inds, assignments, distances = cluster_algo(
            traj, distance_method=distance_method, output=output, **kwargs)

        # de-concatenate the trajectories, assert their validity
        cluster_center_inds = _partition_indices(
            cluster_center_inds, traj_lengths)
        assignments = _partition_list(assignments, traj_lengths)
        assert np.all(assignments >= 0)
        distances = _partition_list(distances, traj_lengths)
        assert np.all(distances >= 0)

        cluster_result = Clustering(
            assignments=assignments,
            distances=distances,
            center_indices=cluster_center_inds)

        # finish timing.
        end = time.clock()
        output.write(
            "Clustering took %s seconds for %s trajectories.\n" %
            ((end - start), len(traj_lst)))

        return cluster_result

    return cluster_fn


def _partition_indices(indices, traj_lengths):
    '''
    Similar to _partition_list in function, this function uses
    `traj_lengths` to determine which 2d trajectory-list index matches
    the given 1d concatenated trajectory index for each index in
    indices.
    '''

    partitioned_indices = []
    for index in indices:
        trj_index = 0
        for traj_len in traj_lengths:
            if traj_len > index:
                partitioned_indices.append((trj_index, index))
                break
            else:
                index -= traj_len
                trj_index += 1

    return partitioned_indices


def _get_distance_method(metric):
    if metric == 'rmsd':
        return md.rmsd
    elif isinstance(metric, str):
        try:
            import msmbuilder.libdistance as libdistance
        except ImportError:
            raise ImproperlyConfigured(
                "To use '{}' as a clustering metric, STAG ".format(metric) +
                "uses MSMBuilder3's libdistance, but we weren't able to " +
                "import msmbuilder.libdistance.")

        def f(X, Y):
            return libdistance.dist(X, Y, metric)
        return f
    elif callable(metric):
        return metric
    else:
        raise ImproperlyConfigured(
            "'{}' is not a recognized metric".format(metric))


def _partition_list(list_to_partition, partition_lengths):
    if np.sum(partition_lengths) != len(list_to_partition):
        raise DataInvalid(
            "List of length {} does not equal lengths to partition {}.".format(
                list_to_partition, partition_lengths))

    partitioned_list = np.full(
        shape=(len(partition_lengths), max(partition_lengths)),
        dtype=list_to_partition.dtype,
        fill_value=-1)

    start = 0
    for num in range(len(partition_lengths)):
        stop = start+partition_lengths[num]
        np.copyto(partitioned_list[num][0:stop-start],
                  list_to_partition[start:stop])
        start = stop

    # this call will mask out all 'invalid' values of partitioned list, in this
    # case all the np.nan values that represent the padding used to make the
    # array square.
    partitioned_list = np.ma.masked_less(partitioned_list, 0, copy=False)

    return partitioned_list
