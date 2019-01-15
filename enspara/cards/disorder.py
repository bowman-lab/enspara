import logging
import numpy as np
from ..util import array as ra

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def transitions(assignments):
    """Computes the frames at which a state transition occurs for a list
    of state assignments.

    Parameters
    ----------
    assignments : array, shape=(n_frames) or (n_frames, n_trjs)

    Returns
    -------
    tt : array, shape=(n_transitions) or RA shape=(n_trjs, n_transitions)
       If the input is one-dimensional, an array of the frames at which
       a state transition occurs. If the input is multidimensional (i.e.
       has more than one "row" or trajectory), a ragged array where each
       row includes the frames at which the transition occurrs. If state
       n and n+1 differ in assignment, the transition is reported as n.

    References
    -------------
    .. [1] Sukrit Singh and Gregory R. Bowman, "Quantifying allosteric communication via 
        both concerted structural changes and conformational disorder with CARDS".
        Journal of Chemical Theory and Computation 2017 13 (4), 1509-1517
        DOI: 10.1021/acs.jctc.6b01181 
    """

    if len(assignments.shape) == 1:
        d = assignments[1:] - assignments[:-1]
        tt = np.where(d != 0)[0]
    else:
        d = assignments[:, 1:] - assignments[:, :-1]
        rows, columns = ra.where(d != 0)
        lengths = np.bincount(rows)
        tt = ra.RaggedArray(columns, lengths=lengths)

    return tt


def traj_ord_disord_times(transition_times):
    """Calculate order, disorder times from a list of the times of transitions.

    Parameters
    ----------
    transition_times : ndarray, shape=(n_transitions,)
        Array containing the timpoints at which transitions happened

    Returns
    -------
    ord_time : float
        The order time
    n_ord : int
        The number of frames in the ordered state
    disord_time : float
        The disorder time
    n_disord
        The number of frames in the disordered state
    """

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

        # ordered time is average waiting time until event from any starting
        # point
        max_waiting_times = [transition_times[0].tolist()] + \
            time_between_events.tolist()
        max_waiting_times = np.array(max_waiting_times)
        sum_waiting_times = max_waiting_times*(max_waiting_times+1.0)/2
        ord_time = sum_waiting_times.sum()/max_waiting_times.sum()

        # time between first and last event counts towards calculation of
        # disordered time
        n_disord = transition_times[-1]-transition_times[0]
        # n_disord = num_transitions-1

        # time until last even counts towards ordered time
        n_ord = transition_times[-1]

    return ord_time, n_ord, disord_time, n_disord


def create_disorder_traj(transition_times, traj_len, ord_time, disord_time):
    # having default to ordered (state 0) as experiment

    num_transitions = transition_times.shape[0]

    # default to ordered (state 0)
    traj = np.zeros(traj_len)

    # first_time = 0
    # last_time = 0
    # no ordered/disordered segments if two few transitions or timescales are
    # too similar
    # GRB left off check about similar ord/disord times because wasn't sure...
    if num_transitions < 2:  # or ord_time < 3*disord_time:
        return traj  # , first_time, last_time
    else:
        # print "assigning"
        # first_time = transition_times[0]
        # last_time = transition_times[-1]
        for i in range(num_transitions-1):
            seg_start = transition_times[i]
            seg_end = transition_times[i+1]
            time_span = seg_end - seg_start
            likelihood_ratio = ord_time/disord_time * np.exp(-time_span*(1./disord_time - 1./ord_time))
            # print "LR", likelihood_ratio
            if likelihood_ratio >= 3.0:  # favors disordered
                traj[seg_start:seg_end] = 1.
            else:
                traj[seg_start:seg_end] = 0.

        return traj  # , first_time, last_time


def assign_order_disorder(rotamer_trajs):
    """Assigns each frame a disordered or ordered state.

    Frames that are ordered will recieve a value of 0, disordered frames
    are assigned 1.

    Parameters
    ----------
    rotamer_trajs: array, shape=(n_features, n_frames)
        Array of rotameric state assignments

    Returns
    -------
    disordered_trajs: list
        List of arrays with disorder/order assignments for each trajectory
    disorder_n_states: ndarray, shape=(n_features,)
        The number of possible states for each feature in disordered_trajs

    References
    ----------
    .. [1] Sukrit Singh and Gregory R. Bowman, "Quantifying allosteric communication via 
        both concerted structural changes and conformational disorder with CARDS".
        Journal of Chemical Theory and Computation 2017 13 (4), 1509-1517
        DOI: 10.1021/acs.jctc.6b01181 
    """

    logger.debug("Calculating ordered/disordered times")
    n_features = rotamer_trajs[0].shape[1]
    transition_times, mean_ordered_times, mean_disordered_times = \
        transition_stats(rotamer_trajs)

    logger.debug("Assigning to disordered states")
    disordered_trajs = []
    for i in range(len(rotamer_trajs)):
        traj_len = rotamer_trajs[i].shape[0]
        dis_traj = np.zeros((traj_len, n_features))
        for j in range(n_features):
            dis_traj[:, j] = create_disorder_traj(
                transition_times[i][j], traj_len, mean_ordered_times[j],
                mean_disordered_times[j])

        disordered_trajs.append(dis_traj.astype('int16'))
    disorder_n_states = 2*np.ones(n_features, dtype='int16')

    return disordered_trajs, disorder_n_states


def transition_stats(rotamer_trajs):
    """Compute the transition time between disordered/ordered states and
    the mean transition time between a set of trajectories' mean tranisiton times

    Parameters
    ----------
    rotamer_trajs: array, shape=(n_features, n_frames)
        Array of rotameric state assignments

    Returns
    -------
    transition_times: list, shape=(n_traj, n_features, variable)
        For each feature in each trajectory, computes the frames at which a state transition
        occurs

    mean_ordered_times: array, shape=(n_features,)
        Mean ordered time for each feature

    mean_disordered_times: array, shape=(n_features,)
        Mean disordered time for each feature

    References
    ----------
    .. [1] Sukrit Singh and Gregory R. Bowman, "Quantifying allosteric communication via 
        both concerted structural changes and conformational disorder with CARDS".
        Journal of Chemical Theory and Computation 2017 13 (4), 1509-1517
        DOI: 10.1021/acs.jctc.6b01181 
    """

    n_traj = len(rotamer_trajs)

    transition_times = []
    n_features = rotamer_trajs[0].shape[1]
    ordered_times = np.zeros((n_traj, n_features))
    n_ordered_times = np.zeros((n_traj, n_features))
    disordered_times = np.zeros((n_traj, n_features))
    n_disordered_times = np.zeros((n_traj, n_features))
    for i in range(n_traj):
        transition_times.append([])
        for j in range(n_features):
            tt = transitions(rotamer_trajs[i][:, j])
            transition_times[i].append(tt)
            (ordered_times[i, j], n_ordered_times[i, j],
             disordered_times[i, j], n_disordered_times[i, j]) = traj_ord_disord_times(tt)

    trj_lengths = np.array([len(a) for a in rotamer_trajs])
    mean_ordered_times = aggregate_mean_times(
        ordered_times, n_ordered_times, trj_lengths)
    mean_disordered_times = aggregate_mean_times(
        disordered_times, n_disordered_times, trj_lengths)

    return transition_times, mean_ordered_times, mean_disordered_times


def aggregate_mean_times(times, n_times, weight):
    """Compute the mean transition time between a set of trajectories'
    mean transition times.

    Parameters
    ----------
    times : array, shape=(n_trajectories, n_features)
        Array of mean transition times for each trajectory and dihedral.
    n_times : array, shape=(n_trajectories, n_features)
        Array of numbers of transitions observed for each trajectory and
        dihedral.
    weight : array, shape=(n_trajectories,)
        Array of weights for each trajectory. Usually used to weight
        trajectories by their length. Any nonnegative weights can be used.

    Returns
    -------
    mean_times : np.ndarray, shape=(n_features,)
        Mean transition time across trajectories for each dihedral.
    """

    n_features = times.shape[1]
    mean_times = np.zeros(n_features)

    # we normalize by the maximum weight, such that the longest trajectory's
    # mean time is unchanged by the calculation.
    nl_weight = weight / np.sum(weight)

    # we suppress divide by zero errors here, since if we never see a
    # transition, the result of the divide by zero (a NaN) is an
    # acceptable representation of that time.
    with np.errstate(all='ignore'):
        for i in range(n_features):
            mean_times[i] = ((times[:, i] * nl_weight).sum())

    return mean_times
