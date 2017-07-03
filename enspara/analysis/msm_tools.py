# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>,
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.

import numpy as np
import scipy.sparse
import warnings
from ..msm.transition_matrices import eq_probs, assigns_to_counts
from ..msm import builders


def KL_divergence(P, Q, base=2):
    """Calculates the Kullbeck-Liebler divergence between two
    probability distributions, P and Q. If P and Q are two dimensional,
    the divergence is calculated between rows. The output is bits by
    default.
    
    Parameters
    ----------
    P : array, shape=(n_distributions, n_values)
        A list of reference probability distributions. i.e. each row
        should sum to one.
    Q : array, shape(n_distributions, n_values)
        A list of query probability distributions. i.e. each row should
        sum to one. Must be the same shape as P.
    base : float, default=2
        The base of the log differences between probability
        distributions. This sets the units of the output. i.e. base=2
        makes the units in bits and base=e makes the units in nats.

    Returns
    ----------
    divergence : array, shape=(n_distributions,)
        The diverences between distributions in P and Q.
    """

    # numpyify distributions
    P = np.array(P)
    Q = np.array(Q)

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
    # disable numpy's warning of log(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        log_likelihoods = P * np.log(P / Q)
    log_likelihoods[np.where(np.isnan(log_likelihoods))] = 0

    # determine axis to sum over
    axis_sum = 0
    if len(P.shape) > 1:
        axis_sum = 1

    # return divergence for each distribution
    divergence = np.sum(log_likelihoods, axis=axis_sum)

    # change units to base
    divergence /= np.log(base)

    return divergence

    
def Q_from_assignments(
        assignments, n_states=None, lag_time=1, builder=builders.normalize,
        prior_counts=None):
    """Generates the reference matrix for relative entropy calculations
       from an assignments matrix.
    """

    # determine prior
    if prior_counts is None:
        total_counts = np.sum([len(ass) - 1 for ass in assignments])
        prior_counts = 1 / total_counts

    # get counts matrix
    Q_counts = assigns_to_counts(
        assignments, n_states=n_states, lag_time=lag_time)

    # add prior counts
    Q_counts = np.array(Q_counts.todense()) + prior_counts

    # compute transition probability matrix
    # disable warning from mle's calculation of eq_probs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, Q_prob, _ = builder(Q_counts, calculate_eq_probs=False)

    return Q_prob

 
def state_relative_entropy(
        P, Q=None, assignments=None, weights=1, state_subset=None,
        base=2.0, **kwargs):
    """The relative entropy between each state in an MSM. For each
    state, i, the relative entropy is calculated as the Kullbeck-Liebler
    divergence between conditional transition probabilities:

    D_KL(P(i)||Q(i)) =  SUM(P(i,j) * log(P(i,j) / Q(i,j)), j)

    where P is the reference probability matrix and Q is the
    probability matrix in question.

    The relative entropy is calculated between P and Q. If Q is not
    supplied, it is calculated from assignments for a particular lagtime
    and symmetrization option. 

    Parameters
    ----------
    P : array, shape=(n_states, n_states)
        The reference transition probability matrix.
    Q : array, shape=(n_states, n_states), default=None 
        A transition probability matrix that diverges from P.
    assignments : array, shape=(n_trajectories, n_frames), default=None
        2D array of trajectories to compute Q from. Note: if assignments
        are provided instead of a probability matrix, details for MSM
        construction are suggested. i.e. lagtime (default of 1),
        symmetrization option (default of none), and prior counts 
        (default of 1/total_counts).
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

    # number of state in MSM
    n_states = P.shape[0]
    if state_subset is None:
        state_subset = ...

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
        2D array of trajectories to compute Q from. Note: if assignments
        are provided instead of a probability matrix, details for MSM
        construction are suggested. i.e. lagtime (default of 1),
        symmetrization option (default of none), and prior counts 
        (default of 1/total_counts).
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

    # determine the number of states from the reference MSM
    n_states = P.shape[0]

    # calculate populations of reference matix if not provided
    if state_subset is None:
        state_subset = ...
    if populations is None:
        populations = eq_probs(P)[state_subset]
        populations /= populations.sum()

    # calculate the KL divergence for each state, weighted by the
    # populations of P
    rel_entropy_mat = state_relative_entropy(
        P, Q=Q, assignments=assignments, weights=populations,
        state_subset=state_subset, base=base, **kwargs)

    # sum over relative entropy mat
    rel_entropy = np.sum(rel_entropy_mat)

    return rel_entropy
