# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numpy as np


def traj_transition_times(state_traj):
    d = state_traj[1:] - state_traj[:-1]
    tt = np.where(d != 0)[0]
    return tt


def traj_ord_disord_times(transition_times):
    # this is for one trajectory
    # n_org and n_disord variables allow weight multiple trajectories

    num_transitions = transition_times.shape[0]

    disord_time = 0.0
    n_disord = 0.0
    ord_time = 0.0
    n_ord = 0.0

    if num_transitions == 1:
        waiting_time = transition_times[0]
        n_ord = waiting_time
        ord_time = waiting_time*(waiting_time+1.0)/2
    elif num_transitions > 1:
        time_between_events = np.diff(transition_times)

        # disordered time is average waiting time between events
        disord_time = time_between_events.mean()

        # ordered time is average waiting time until event from any starting point
        max_waiting_times = [transition_times[0]] + time_between_events.tolist()
        max_waiting_times = np.array(max_waiting_times)
        sum_waiting_times = max_waiting_times*(max_waiting_times+1.0)/2
        ord_time = sum_waiting_times.sum()/max_waiting_times.sum()

        # time between first and last event counts towards calculation of disordered time
        n_disord = transition_times[-1]-transition_times[0]
        #n_disord = num_transitions-1
        
        # time until last even counts towards ordered time
        n_ord = transition_times[-1]

    return ord_time, n_ord, disord_time, n_disord
    
def create_disorder_traj(transition_times, traj_len, ord_time, disord_time):
    # having default to ordered (state 0) as experiment

    num_transitions = transition_times.shape[0]
    
    # default to ordered (state 0)
    traj = np.zeros(traj_len)
    
    # if don't see any transitions, could be infinitely long until one occurs...
    #traj[:] = np.inf
    
    #first_time = 0
    #last_time = 0
    # no ordered/disordered segments if two few transitions or timescales are too similar
    # GRB left off check about similar ord/disord times because wasn't sure...    
    if num_transitions < 2: # or ord_time < 3*disord_time:
        #print "No distinction"
        return traj #, first_time, last_time
    else:
        #print "assigning"
        #first_time = transition_times[0]
        #last_time = transition_times[-1]
        for i in range(num_transitions-1):
            seg_start = transition_times[i]
            seg_end = transition_times[i+1]
            time_span = seg_end - seg_start
            likelihood_ratio = ord_time/disord_time * np.exp(-time_span*(1./disord_time - 1./ord_time))
            #print "LR", likelihood_ratio
            if likelihood_ratio >= 3.0: # favors disordered
                traj[seg_start:seg_end] = 1.
            else:
                traj[seg_start:seg_end] = 0.

        return traj #, first_time, last_time
        
    
    