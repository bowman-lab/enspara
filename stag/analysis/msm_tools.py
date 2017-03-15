# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>,
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import msmbuilder.msm
import numpy as np
import scipy.sparse
from ..msm.transition_matrices import eq_probs

def generate_sparse_mat(mat):
    rows,cols = np.nonzero(mat)
    data = mat[rows,cols]
    sparse_mat = scipy.sparse.coo_matrix(
        (data, (rows, cols)), shape=mat.shape)
    return sparse_mat

def generate_mapped_matrix(mat, mapping, n_states=None):
    """returns a sparse matrix of the transition probability matrix from a
       mapping dictionary. mapping dictionary is assumed to be the output of
       msmbuilder3's msm.mapping_
    """
    if scipy.sparse.issparse(mat):
        rows = mat.row
        cols = mat.col
        data = mat.data
    else:
        rows,cols = np.nonzero(mat)
        data = mat[rows, cols]
    keys = np.array(list(mapping.keys()))
    new_rows = keys[rows]
    new_cols = keys[cols]
    if n_states is None:
        n_states = keys.max() + 1 # this is plus 1 because of zero index
    mat_shape = (n_states, n_states)
    mat_new = scipy.sparse.coo_matrix(
        (data, (new_rows, new_cols)), shape=mat_shape)
    return mat_new

def Q_from_assignments(
        assignments, n_states, lagtime=1, msm_type='none',
        trimming='off'):
    """Generates the reference matrix for relative entropy calculations
       from an assignments matrix.
    """
    prior_counts = 1/float(n_states)
    # get counts from msmbuilder (TODO: update to stag stuff)
    msm_data = msmbuilder.msm.MarkovStateModel(
        lag_time=lagtime, reversible_type=msm_type, ergodic_cutoff=trimming,
        verbose=False)
    msm_data.fit(assignments)
    Q_counts = msm_data.countsmat_
    mapping = msm_data.mapping_
    # convert count matrix
    Q_counts_full = generate_mapped_matrix(Q_counts, mapping, n_states=n_states)
    Q_counts_new = np.array(Q_counts_full.todense()) + prior_counts
    # Transpose probability matrix (TODO: update types of symmetrization)
    Q_new = Q_counts_new + Q_counts_new.T
    Q_new = Q_new/Q_new.sum(axis=1)[:,None]
    return Q_new
 
def relative_entropy_matrix(
        P, Q=None, assignments=None, populations=None, state_subset=None):
    """Returns the matrix of Dij defined in the relative_entropy function
       If Q is not specified, it will be calculated from assignments.
       TODO: make stag's assignment generation faster so we don't have to
       rely on msmbuilder
    """
    # number of state in MSM
    n_states = P.shape[0]
    # check inputs
    if (Q is None) and (assignments is None):
        print('must specify Q or calculate Q from assignments')
    elif (Q is None):
        Q = Q_from_assignments(assignments, n_states)
    # ensure correct shape and numbering of transition matricies
    if (P.shape != Q.shape):
        raise
    # calculate populations of reference matix if not provided
    if populations is None:
        populations = eq_probs(P)
    if state_subset is None:
        state_subset = np.arange(n_states)
    # obtain relative entropy matrix
    rel_entropy_mat = populations[:,None]*P*np.log10(P/Q)
    # make nan entries zero (if Pij = 0, Dij=0)
    iis_nan = np.where(np.isnan(rel_entropy_mat))
    rel_entropy_mat[iis_nan] = 0
    rel_entropy_mat = rel_entropy_mat[state_subset, :]
    rel_entropy_mat = rel_entropy_mat[:, state_subset]
    return rel_entropy_mat

def relative_entropy(
        P, Q=None, assignments=None, populations=None, state_subset=None):
    """The Kullback-leibler divergence, defined as:

       D_KL(P||Q) = sum(Dij)

       Dij = P(i) * P(i,j) * log(P(i,j) / Q(i,j))

       where P is the reference probability matrix and Q is the
       probability matrix in question.
    """
    rel_entropy_mat = relative_entropy_matrix(
        P, Q=Q, assignments=assignments, populations=populations,
        state_subset=state_subset)
    rel_entropy = np.sum(rel_entropy_mat)
    return rel_entropy
