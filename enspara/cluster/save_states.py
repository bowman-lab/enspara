import logging

import gc
import glob
import mdtraj as md
import numpy as np
import os
import time
import subprocess as sp
from multiprocessing import Pool


def _save_states(centers_info):
    states = centers_info['state']
    confs = centers_info['conf']
    frames = centers_info['frame']
    traj_filename = centers_info['trj_filename'][0]
    output_directory = centers_info['output'][0]
    topology = centers_info['topology'][0]
    traj = md.load(traj_filename, top=topology)
    for num in range(len(states)):
        pdb_filename = "{dir}State{state}-{conf}.pdb".format(
            dir=output_directory, state=states[num], conf=confs[num])
        center = traj[frames[num]]
        center.save_pdb(pdb_filename)


def unique_states(assignments):
    '''
    Search assignments array and return a list of the state ids within.
    '''

    state_nums = np.unique(assignments)
    state_nums = state_nums[np.where(state_nums != -1)]
    return state_nums


def save_states(
        assignments, distances, state_nums=None,
        traj_filenames='./Trajectories/*.xtc',
        output_directory='./PDBs/', topology='prot_masses.pdb',
        largest_center=np.inf, n_confs=1, n_processes=1, verbose=True):
    '''
    Saves specified state-numbers by searching through the assignments and
    distances. Can specify a largest distance to a cluster center to save
    computational time searching for min distances. If multiple conformations
    are saved, the center is saved as conf-0 and the rest are random
    conformations.
    '''

    t0 = time.time()
    if state_nums is None:
        state_nums = unique_states(assignments)

    # Get full pathway to input and output directories and ensure that they
    # exist
    if type(traj_filenames) == str:
        traj_filenames = np.array(
            [os.path.abspath(trj) for trj in glob.glob(traj_filenames)])
    output_directory = os.path.abspath(output_directory)+"/"
    if not os.path.exists(output_directory):
        sp.check_call(["mkdir", output_directory])
    # reduce the number of conformations to search through
    reduced_iis = np.where((distances>-0.1)*(distances < largest_center))
    reduced_assignments = assignments[reduced_iis]
    reduced_distances = distances[reduced_iis]
    centers_location = []
    for state in state_nums:
        state_iis = np.where(reduced_assignments == state)
        nconfs_in_state = len(state_iis[0])
        if nconfs_in_state >= n_confs:
            center_picks = np.array([0])
            if n_confs > 1:
                center_picks = np.append(
                    center_picks,
                    np.random.choice(
                        range(1, nconfs_in_state), n_confs-1, replace=False))
        else:
            center_picks = np.array([0])
            center_picks = np.append(
                center_picks, np.random.choice(nconfs_in_state, n_confs - 1))
        state_centers = np.argsort(reduced_distances[state_iis])[center_picks]
        # Obtain information on conformation locations within trajectories
        traj_locations = reduced_iis[0][state_iis[0][state_centers]]
        frame_nums = reduced_iis[1][state_iis[0][state_centers]]
        for conf_num in range(n_confs):
            traj_num = traj_locations[conf_num]
            centers_location.append(
                (
                    state, conf_num, traj_num,
                    frame_nums[conf_num], traj_filenames[traj_num],
                    output_directory, topology))
    if type(topology) == str:
        centers_location = np.array(
            centers_location, dtype=[
                ('state', 'int'), ('conf', 'int'), ('traj_num', 'int'),
                ('frame', 'int'), ('trj_filename', np.str_, 500),
                ('output', np.str_, 500),
                ('topology', np.str_, 500)])
    else:
        centers_location = np.array(
            centers_location, dtype=[
                ('state', 'int'), ('conf', 'int'), ('traj_num', 'int'),
                ('frame', 'int'), ('trj_filename', np.str_, 500),
                ('output', np.str_, 500),
                ('topology', type(topology))])
    unique_trajs = np.unique(centers_location['traj_num'])
    partitioned_centers_info = []
    for traj in unique_trajs:
        partitioned_centers_info.append(
            centers_location[np.where(centers_location['traj_num'] == traj)])

    logging.debug("  Saving states!")

    pool = Pool(processes=n_processes)
    pool.map(_save_states, partitioned_centers_info)
    pool.terminate()
    gc.collect()

    t1 = time.time()
    logging.debug("    Finished in "+str(t1-t0)+" sec")
