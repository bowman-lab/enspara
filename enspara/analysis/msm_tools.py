# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>,
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.

import numpy as np
import scipy.sparse
from ..msm.transition_matrices import eq_probs, assigns_to_counts
from ..msm import builders


def KL_divergence(P, Q, base=2):
    """Calculates the Kullbeck-Liebler divergence between two
    probability distributions, P and Q. If P and Q are two dimensional,
    the divergence is calculated between rows. The output is bits by
    default.
    """
    # Check shape of distributions
    if P.shape != Q.shape:
        raise

    # Ensure non-negative probabilties
    if len(np.where(P < 0)[0]) > 0:
        raise
    if len(np.where(Q < 0)[0]) > 0:
        raise

    # calculate inners and set nans to zero (this is okay because
    # xlogx = 0 @ x=0)
    log_likelihoods = P * np.log(P / Q)
    log_likelihoods[np.where(np.isnan(log_likelihoods))] = 0

    # determine axis to sum over
    axis_sum = 0
    if len(P.shape) > 0:
        axis_sum = 1

    # return divergence for each distribution
    divergence = np.sum(log_likelihoods, axis=axis_sum)

    # change units to base
    divergence /= np.log(base)
    return divergence

    
def Q_from_assignments(
        assignments, n_states, lag_time=1, builder=builders.normalize,
        prior_counts=None):
    """Generates the reference matrix for relative entropy calculations
       from an assignments matrix.
    """
    # determine prior
    if prior_counts is None:
        prior_counts = 0
    # get counts matrix
    Q_counts = assigns_to_counts(
        assignments, n_states=n_states, lag_time=lag_time)
    # add prior counts
    Q_counts = np.array(Q_counts.todense()) + prior_counts
    # compute transition probability matrix
    _, Q_prob, _ = builder(Q_counts, calculate_eq_probs=False)
    return Q_prob

 
def state_relative_entropy(
        P, Q=None, assignments=None, weights=1, state_subset=None,
        base=2.0, **kwargs):
    """Returns the matrix of Dij defined in the relative_entropy function
       If Q is not specified, it will be calculated from assignments.
    """
    # number of state in MSM
    n_states = P.shape[0]
    # check inputs
    if (Q is None) and (assignments is None):
        print('must specify Q or calculate Q from assignments')
    elif (Q is None):
        Q = Q_from_assignments(
            assignments, n_states=n_states, **kwargs)
    # obtain relative entropy matrix
    rel_entropy_mat = KL_divergence(P, Q, base=base)
    return rel_entropy_mat[state_subset]*weights


def relative_entropy(
        P, Q=None, assignments=None, populations=None, state_subset=None,
        base=2.0, **kwargs):
    """The relative entropy between MSMs defined as:

    D_MSM(P||Q) = sum(Dij)

    Dij = P(i) * P(i,j) * log(P(i,j) / Q(i,j))

    where P is the reference probability matrix and Q is the
    probability matrix in question.

    The relative entropy is calculated between P and Q. If Q is not
    supplied, it is calculated from assignments for a particular lagtime
    and symmetrization option. If populations of P are not supplied,
    they are calculated.

    Parameters
    ----------
    P : array, shape=(n_states, n_states)
        The reference transition probability matrix.
    Q : array, shape=(n_states, n_states), default=None 
        A transition probability matrix that diverges from P.
    assignments : array, shape=(n_trajectories, n_frames), default=None
        2D array of trajectories to compute Q from.
    lag_time : int, default=1
        The lagtime to compute Q with from assignments.
    builder : function, default=builder.normalize
        The enspara builder function used for symmetrization of count
        matricies.
    prior_counts : float, default=None
        The prior counts to add to the count matrix when computing Q
        from assignments. This is essential if there are states that
        have not been visited.
    populations : array, shape=(n_states,), default=None
        The equilibrium populations of the reference MSM. If not
        supplied, this is calculated from the eigenspectrum of P.
    state_subset : array, shape=(n_subset,), default=None
        Optionally specify a subset of states to use for calculation of
        relative entropy. These states will be the only ones that
        contribute to the MSMs relative entropy value. If no subset is
        specified, all states will be used.
    base : float, default=2.0
        The base used for calculation of Kullbeck-Liebler divergence.
        This specified the units of the output, i.e. base 2 will be in
        bits and base e will be in nats.
    """
    n_states = P.shape[0]
    # calculate populations of reference matix if not provided
    if state_subset is None:
        state_subset = np.arange(n_states)
    if populations is None:
        populations = eq_probs(P)[state_subset]
        populations /= populations.sum()
    # calculate the KL divergence for each state
    rel_entropy_mat = state_relative_entropy(
        P, Q=Q, assignments=assignments, weights=populations,
        state_subset=state_subset, base=base, **kwargs)
    rel_entropy = np.sum(rel_entropy_mat)
    return rel_entropy
