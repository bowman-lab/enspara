# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import warnings

import numpy as np

from .. import exception
from ..msm import builders
from ..msm.transition_matrices import eq_probs, assigns_to_counts


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
        assignments, max_n_states=n_states, lag_time=lag_time)

    # add prior counts
    Q_counts = np.array(Q_counts.todense()) + prior_counts

    # compute transition probability matrix
    # disable warning from mle's calculation of eq_probs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, Q_prob, _ = builder(Q_counts, calculate_eq_probs=False)

    return Q_prob


def relative_entropy_per_state(
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
        state_subset = Ellipsis

    # check inputs
    if (Q is None) and (assignments is None):
        print('must specify Q or calculate Q from assignments')
    elif (Q is None):
        Q = Q_from_assignments(
            assignments, n_states=n_states, **kwargs)

    # obtain relative entropy matrix
    rel_entropy_mat = kl_divergence(P, Q, base=base)

    return rel_entropy_mat[state_subset]*weights


def relative_entropy_msm(
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

    # calculate populations of reference matix if not provided
    if state_subset is None:
        state_subset = Ellipsis
    if populations is None:
        populations = eq_probs(P)[state_subset]
        populations /= populations.sum()

    # calculate the KL divergence for each state, weighted by the
    # populations of P
    rel_entropy_mat = relative_entropy_per_state(
        P, Q=Q, assignments=assignments, weights=populations,
        state_subset=state_subset, base=base, **kwargs)

    # sum over relative entropy mat
    rel_entropy = np.sum(rel_entropy_mat)

    return rel_entropy


def energy_to_probability(u, kT=2.479):
    p = np.exp(-(u-u.mean())/kT)
    p /= p.sum()
    return p


def shannon_entropy(p, normalize=True):
    """Compute the Shannon entropy of a uni- or multi-variate
    distribution.

    Parameters
    ----------
    p : np.ndarray
        Vector or matrix of probabilities representing the (potentially
        multivariate) distribution over which to calculate the entropy.
    normalize : bool, default=True
        Forcibly normalize the sum of p to one. (Not in place;
        duplicates p.)

    Returns
    -------
    H : float
        The Shannon entropy of the distribution
    """

    if normalize:
        p = np.copy(p) / np.sum(p)

    H = -np.sum(p * np.log(p, where=(p > 0)))

    return H


def kl_divergence(P, Q, base=2):
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
    for M in (P, Q):
        if len(np.where(M < 0)[0]) > 0:
            raise exception.DataInvalid(
                'The supplied matrix contained a negative probability:\n%s' %
                M)

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


def js_divergence(p, q):
    m = 0.5*(p+q)
    js = 0.5 * kl_divergence(p, m) + 0.5*kl_divergence(q, m)
    return js
