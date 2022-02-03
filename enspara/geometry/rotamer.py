import mdtraj as md
import numpy as np
from enspara.exception import DataInvalid


def dihedral_angles(traj, dihedral_type):
    valid_dihedral_types = ["phi", "psi", "chi1", "chi2", "chi3", "chi4"]
    if dihedral_type not in valid_dihedral_types:
        return None, None, None

    f = getattr(md, "compute_%s" % dihedral_type)
    atom_inds, angles = f(traj)

    # transform so angles range from 0 to 360 instead of radians or -180 to 180
    angles = np.rad2deg(angles)
    angles[np.where(angles < 0)] += 360

    n_angles = angles.shape[1]
    ref_atom_inds = np.zeros(n_angles)
    for i in range(n_angles):
        atom = traj.topology.atom(atom_inds[i, 2])
        ref_atom_inds[i] = int(atom.index)

    return angles, atom_inds


def _rotamers(angles, hard_boundaries, buffer_width=15):
    """Rotamer state assignment for any trajectory of dihedral angles
    using a buffered transiton approach.

    NOTE: This method works entirely in degrees and assumes that you have
    already transformed your data to span from 0 to 360

    Parameters
    ----------
    angles : array-like, shape=(n_frames)
        Time-series data containing the values of
        a single dihedral angle from a trajectory. This array MUST:
        a) Be passed in degrees
        b) Span from 0 to 360 in range
    hard_boundaries : array-like, shape=(n_boundaries)
        Single set of numbers containing the "hard" boundaries denoting
        rotamer basins. This array MUST contain 0 as the first value and
        360 as the last value.
    buffer_width : int, default=15
        Size (in degrees) of the buffer region on either side of the
        rotamer barrier; a value of 0 indicates no buffer.

    Returns
    --------
    rotamers : array-like, shape=(n_frames)
        Time-series data with rotamer state assignments. Each element i
        contains the rotamer state assignment computed using the "angles"
        array.

    See Also
    --------
    is_buffered_transition, get_gates
    """
    n_basins = len(hard_boundaries) - 1

    if buffer_width < 0 or buffer_width >= 360. / n_basins:
        raise DataInvalid('Buffer width (got %s) must be between 0 and '
                          '360 degrees.' % buffer_width)
    if hard_boundaries[0] != 0 or hard_boundaries[-1] != 360:
        raise DataInvalid('hard_boundaries list must start with 0 and '
                          'end with 360, list was %s.' % hard_boundaries)

    # Need to establish how long it is and create an appropriate output array
    n_frames = len(angles)
    rotamers = -1 * np.ones(n_frames, dtype='int16')

    # First, assign the first state to its rotamer bin
    # make sure assign first frame
    for i in range(n_basins):
        if angles[0] < hard_boundaries[i + 1]:
            rotamers[0] = i
            break

    # Now we will go through each subsequent element of the array and
    # assign each state based on whether or not there is a buffered transition
    cur_state = rotamers[0]
    for i in range(1, n_frames):
        new_angle = angles[i]
        cur_angle = angles[i - 1]

        # If there is a buffered transition we will reassign states
        if is_buffered_transition(cur_state, new_angle, hard_boundaries,
                                  buffer_width):
            cur_state = np.digitize(new_angle, hard_boundaries) - 1

        rotamers[i] = cur_state

    return rotamers


def is_buffered_transition(cur_state, new_angle, hard_boundaries,
                           buffer_width):
    """Returns whether or not the change in angle is a buffered transition.
    A "buffered transition" is defined as a transition between rotameric states
    that happens across a buffer-zone. That is, the transition occurred while
    the rotamer is buffered.

    Parameters
    ----------
    cur_state : int,
        Rotameric basin index presently being occupied by the dihedral
    new_angle : float,
        Dihedral angle value at the dihedral's subsequent timestep. Note that
        this value must meet the same two conditions as described above in
        _rotamers method.
    hard_boundaries : array-like, shape=(n_boundaries)
        Single set of numbers containing the "hard" boundaries denoting
        rotamer basins. This array MUST contain 0 as the first value and
        360 as the last value.
    buffer_width : int, default=15
        size of the buffer region on either side of the
        rotamer barrier.

    Returns
    ---------
    result : boolean
        Returns a boolean representing whether or not the transition from
        cur_state to new_angle represents a real transiton out of a
        buffer zone


    See Also
    --------
    _rotamers, get_gates
    """

    # By default, we assume that no transition has occurred
    result = False

    # Given the current angle, we need to identify the "gates"
    lower_bound, upper_bound = get_gates(cur_state, hard_boundaries,
                                         buffer_width)

    # Keep in mind these are gates representing what new_angle has to EXIT.

    # This means that if new_angle is within the interval spanning these gate
    # it has transitioned.

    # We can check to see if this is a wrap around state or not.

    # If it is  meant to be a "wrap around", then the
    # difference (Upper - Lower) would be negative because gates are
    # flipped.
    if (upper_bound < lower_bound):
        if (upper_bound <= new_angle <= lower_bound):
            result = True

    # If the difference is positive, then we just need to flip our inequality
    if (upper_bound > lower_bound):
        if (not (lower_bound <= new_angle <= upper_bound)):
            result = True

    return result


def get_gates(cur_state, hard_boundaries, buffer_width):
    """Obtains the gates that represent the edges that a dihedral must exit
    from to undergo a buffered transition.

    Parameters
    -----------
    cur_state : int,
        Rotameric basin index presently being occupied by the dihedral
    hard_boundaries : array-like, shape=(1, n_boundaries)
        Single set of numbers containing the "hard" boundaries denoting
        rotamer basins. This array MUST contain 0 as the first value and
        360 as the last value.
    buffer_width : int, default=15
        size of the buffer region on either side of the
        rotamer barrier.

    Returns
    ----------
    lower_bound : int
        Lower gate that must be crossed for a transition to occur
    upper_bound : int
        Upper gate that must be crossed for a transition to occur

    See Also
    --------
    _rotamers, is_buffered_transition

    """
    # First, assign the current angle into one of the boundaries
    n_basins = len(hard_boundaries) - 1
    state_num = int(cur_state)

    # for i in range(n_basins):
    #     if cur_angle < hard_boundaries[i+1]:
    #         state_num = i
    #         break

    # Now that we know the state it's in - we can dictate it's gates
    lower_bound = hard_boundaries[state_num]
    upper_bound = hard_boundaries[state_num+1]
    # These represents the edges which have to be crossed by new_angle

    # If the lower bound is zero, set upper bound to 360 (wrap around)
    # If the upper bound is 360, set lower bound to 0 (wrap around)
    if (lower_bound == 0):
        lower_bound = 360
    if (upper_bound == 360):
        upper_bound = 0

    lower_bound -= buffer_width
    upper_bound += buffer_width

    # Keep in mind that these are gates representing what boundaries must be
    # crossed by the new angle to be considered a transition
    # This point will be addressed in the is_buffered_transition method

    return lower_bound, upper_bound


def phi_rotamers(traj, buffer_width=15):
    hard_boundaries = [0, 180, 360]
    angles, atom_inds = dihedral_angles(traj, 'phi')

    n_frames, n_angles = angles.shape
    rotamers = np.zeros((n_frames, n_angles), dtype='int16')
    for i in range(n_angles):
        rotamers[:, i] = _rotamers(angles[:, i], hard_boundaries, buffer_width)

    n_states = 2*np.ones(n_angles, dtype='int16')

    return rotamers, atom_inds, n_states


def psi_rotamers(traj, buffer_width=15):
    angles, atom_inds = dihedral_angles(traj, 'psi')

    # shift by 100 so boundaries at 0 and 360
    shifted_angles = angles-100
    shifted_angles[np.where(shifted_angles < 0)] += 360
    hard_boundaries = [0, 160, 360]

    n_frames, n_angles = angles.shape
    rotamers = np.zeros((n_frames, n_angles), dtype='int16')
    for i in range(n_angles):
        rotamers[:, i] = _rotamers(shifted_angles[:, i], hard_boundaries,
                                   buffer_width)

    n_states = 2*np.ones(n_angles, dtype='int16')

    return rotamers, atom_inds, n_states


def chi_rotamers(traj, buffer_width=15):
    # could make a dictionary of boundaries for different residue/dihedral
    # types
    hard_boundaries = [0, 120, 240, 360]

    angles, atom_inds = dihedral_angles(traj, 'chi1')
    for i in range(2, 5):
        more_angles, more_atom_inds = dihedral_angles(traj, 'chi%d' % i)
        angles = np.append(angles, more_angles, axis=1)
        atom_inds = np.append(atom_inds, more_atom_inds, axis=0)

    n_frames, n_angles = angles.shape
    rotamers = np.zeros((n_frames, n_angles), dtype='int16')
    for i in range(n_angles):
        rotamers[:, i] = _rotamers(angles[:, i], hard_boundaries, buffer_width)

    n_states = 3*np.ones(n_angles, dtype='int16')

    return rotamers, atom_inds, n_states


def all_rotamers(traj, buffer_width=15):
    """Compute the rotameric states of a trajectory over time.

    Parameters
    ----------
    traj : md.Trajectory
        Trajectory from which to compute a rotamer trajectory.
    buffer_width: int, default=15
        Width of the "no-man's land" between rotameric bins in which no
        assignment is made.

    Returns
    -------
    all_rotamers : np.ndarray, shape=(n_frames, n_dihedrals)
        Assignment of each dihedral to a rotameric state as an int in
        the range 0-2.
    all_atom_inds : np.ndarray, shape=(n_dihedrals, 4)
        Array of the four atom indices that define each dihedral angle.
    all_n_states : np.ndarray, shape=(n_dihedrals,)
        Array indicating the maximum number of states a dihedral angle
        is expected to take (as a consequence of its topology); this
        value differs for backbone and sidechain dihedrals.

    References
    ----------
    .. [1] Sukrit Singh and Gregory R. Bowman, "Quantifying allosteric communication via
        both concerted structural changes and conformational disorder with CARDS".
        Journal of Chemical Theory and Computation 2017 13 (4), 1509-1517
        DOI: 10.1021/acs.jctc.6b01181
    """
    phi_rotameric_states, phi_atom_inds, n_phi_states = phi_rotamers(
        traj, buffer_width=buffer_width)
    all_rotamers, all_atom_inds, all_n_states = phi_rotameric_states, \
        phi_atom_inds, n_phi_states

    psi_rotameric_states, psi_atom_inds, n_psi_states = psi_rotamers(
        traj, buffer_width=buffer_width)
    all_rotamers = np.append(all_rotamers, psi_rotameric_states, axis=1)
    all_atom_inds = np.append(all_atom_inds, psi_atom_inds, axis=0)
    all_n_states = np.append(all_n_states, n_psi_states, axis=0)

    chi_rotameric_states, chi_atom_inds, n_chi_states = chi_rotamers(
        traj, buffer_width=buffer_width)
    all_rotamers = np.append(all_rotamers, chi_rotameric_states, axis=1)
    all_atom_inds = np.append(all_atom_inds, chi_atom_inds, axis=0)
    all_n_states = np.append(all_n_states, n_chi_states, axis=0)

    assert issubclass(all_rotamers.dtype.type, np.integer)
    assert issubclass(all_n_states.dtype.type, np.integer)

    return all_rotamers, all_atom_inds, all_n_states
