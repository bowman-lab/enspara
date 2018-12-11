"""An implementation of the Baysean Agglomerative Clustering Engine.
"""

import logging
import functools
import multiprocessing

import numpy as np

import scipy
import scipy.io
import scipy.sparse

from enspara import exception

logger = logging.getLogger(__name__)


def getInds(c, stateInds, chunkSize, updateSingleState=None):
    indices = []
    for s in stateInds:
        if scipy.sparse.issparse(c):
            dest = np.where(c[s, :].toarray()[0] > 1)[0]
        else:
            dest = np.where(c[s, :] > 1)[0]
        if updateSingleState is not None:
            dest = dest[np.where(dest != updateSingleState)[0]]
        else:
            dest = dest[np.where(dest > s)[0]]
        if dest.shape[0] == 0:
            continue
        elif dest.shape[0] < chunkSize:
            indices.append((s, dest))
        else:
            i = 0
            while dest.shape[0] > i:
                if i+chunkSize > dest.shape[0]:
                    indices.append((s, dest[i:]))
                else:
                    indices.append((s, dest[i:i+chunkSize]))
                i += chunkSize
    return indices


def bace(c, n_macrostates, chunk_size=100, n_procs=1):
    """Perform baysean agglomerative coarse-graining procedure (BACE)

    If you use this code, you should read and cite [1]_.

    Parameters
    ----------
    c : array-like, shape=(n_states, n_states)
        Transition counts matrix to perform agglomeration on
    n_macrostates : int
        Number of macrostates to coarse-grain into.
    n_procs : int, default=1
        Number of parallel processes to use.
    chunk_size : int, default=100

    Returns
    -------
    bayes_factors : dict
        Mapping from number of macrostates to the bayes' factor of that
        lumping.
    labels : dict
        Mapping from number of macrostates to the labelling of
        microstates into that number of macrostates.


    References
    ----------
    .. [1] Bowman, G. R. Improved coarse-graining of Markov state models via
        explicit consideration of statistical uncertainty. J Chem Phys 137,
        134111 (2012).
    """

    # perform filter
    logger.info("Checking for states with insufficient statistics")
    c, state_map, statesKeep = baysean_prune(c, n_procs)
    c = c.astype('float')
    logger.info("Merged %d states with insufficient statistics into their "
                "kinetically-nearest neighbor", c.shape[0] - len(statesKeep))

    # get num counts in each state (or weight)
    w = np.array(c.sum(axis=1)).flatten()
    w[statesKeep] += 1

    unmerged = np.zeros(w.shape[0], dtype=np.int8)
    unmerged[statesKeep] = 1

    # get nonzero indices in upper triangle
    indRecalc = getInds(c, statesKeep, chunk_size)
    if scipy.sparse.issparse(c):
        dMat = scipy.sparse.lil_matrix(c.shape)
    else:
        dMat = np.zeros(c.shape, dtype=np.float32)

    if scipy.sparse.issparse(c):
        c = c.tocsr()

    bayes_factors = {}
    labels = {}

    dMat, minX, minY = calcDMat(c, w, bayes_factors, indRecalc, dMat, n_procs,
                                statesKeep, unmerged, chunk_size)
    logger.info("Coarse-graining...")

    for cycle in range(c.shape[0] - n_macrostates):
        logger.info("Iteration %d, merging %d states",
                    cycle, c.shape[0] - cycle)
        rslt = mergeTwoClosestStates(
            c, w, bayes_factors, indRecalc, dMat, n_procs, state_map,
            statesKeep, minX, minY, unmerged, chunk_size)
        c, w, indRecalc, dMat, state_map, statesKeep, unmerged, minX, \
            minY = rslt

        labels[c.shape[0] - cycle - 1] = state_map.astype(int)

    return bayes_factors, labels


def mergeTwoClosestStates(
        c, w, bayes_factors, indRecalc, dMat, nProc, state_map, statesKeep,
        minX, minY, unmerged, chunkSize):
    sparse = scipy.sparse.issparse(c)
    if sparse:
        c = c.tolil()
    if unmerged[minX]:
        c[minX, statesKeep] += unmerged[statesKeep] / c.shape[0]
        unmerged[minX] = 0
        if sparse:
            c[statesKeep, minX] += (
                np.matrix(unmerged[statesKeep]).transpose() / c.shape[0])
        else:
            c[statesKeep, minX] += unmerged[statesKeep] / c.shape[0]
    if unmerged[minY]:
        c[minY, statesKeep] += unmerged[statesKeep] / c.shape[0]
        unmerged[minY] = 0
        if sparse:
            c[statesKeep, minY] += (
                np.matrix(unmerged[statesKeep]).transpose() / c.shape[0])
        else:
            c[statesKeep, minY] += unmerged[statesKeep] / c.shape[0]
    c[minX, statesKeep] += c[minY, statesKeep]
    c[statesKeep, minX] += c[statesKeep, minY]
    c[statesKeep, minY] = c[minY, statesKeep] = 0
    dMat[minX, :] = dMat[:, minX] = 0
    dMat[minY, :] = dMat[:, minY] = 0

    if sparse:
        c = c.tocsr()
    w[minX] += w[minY]
    w[minY] = 0
    statesKeep = statesKeep[np.where(statesKeep != minY)[0]]
    indChange = np.where(state_map == state_map[minY])[0]
    state_map = renumberMap(state_map, state_map[minY])
    state_map[indChange] = state_map[minX]
    indRecalc = getInds(c, [minX], chunkSize, updateSingleState=minX)
    dMat, minX, minY = calcDMat(c, w, bayes_factors, indRecalc, dMat, nProc,
                                statesKeep, unmerged, chunkSize)
    return c, w, indRecalc, dMat, state_map, statesKeep, unmerged, minX, minY


def renumberMap(state_map, stateDrop):
    for i in range(state_map.shape[0]):
        if state_map[i] >= stateDrop:
            state_map[i] -= 1
    return state_map


def calcDMat(c, w, bayes_factors, indices, dMat, n_procs, statesKeep,
             unmerged, chunkSize):
    n = len(indices)
    if n > 1 and n_procs > 1:
        n_procs = min(n, n_procs)
        step = n // n_procs

        end_index = n if n % step > 3 else n-step
        dlims = zip(range(0, end_index, step),
                    list(range(step, end_index, step)) + [n])

        with multiprocessing.Pool(processes=n_procs) as pool:
            result = pool.map(
                functools.partial(multiDist, c=c, w=w, statesKeep=statesKeep,
                                  unmerged=unmerged, chunkSize=chunkSize),
                [indices[start:stop] for start, stop in dlims])

            d = np.vstack(result)
    else:
        d = multiDist(indices, c, w, statesKeep, unmerged, chunkSize)
    for i in range(len(indices)):
        dMat[indices[i][0], indices[i][1]] = d[i][:len(indices[i][1])]

    # BACE BF inverted so can use sparse matrices
    if scipy.sparse.issparse(dMat):
        minX = minY = -1
        maxD = 0
        for x in statesKeep:
            if len(dMat.data[x]) == 0:
                continue
            pos = np.argmax(dMat.data[x])
            if dMat.data[x][pos] > maxD:
                maxD = dMat.data[x][pos]
                minX = x
                minY = dMat.rows[x][pos]
    else:
        indMin = dMat.argmax()
        minX = int(np.floor(indMin / dMat.shape[1]))
        minY = indMin % dMat.shape[1]

    bayes_factors[statesKeep.shape[0]-1] = 1./dMat[minX, minY]

    return dMat, minX, minY


def multiDist(indicesList, c, w, statesKeep, unmerged, chunkSize):
    d = np.zeros((len(indicesList), chunkSize), dtype=np.float32)
    for j in range(len(indicesList)):
        indices = indicesList[j]
        ind1 = indices[0]

        if scipy.sparse.issparse(c):
            c1 = (c[ind1, statesKeep].toarray()[0] + unmerged[ind1] *
                  unmerged[statesKeep] / c.shape[0])
        else:
            c1 = (c[ind1, statesKeep] + unmerged[ind1] * unmerged[statesKeep] /
                  c.shape[0])

        # BACE BF inverted so can use sparse matrices
        d[j, :indices[1].shape[0]] = 1 / multiDistHelper(
            indices[1], c1, w[ind1], c, w, statesKeep, unmerged)
    return d


def multiDistHelper(indices, c1, w1, c, w, statesKeep, unmerged):
    d = np.zeros(indices.shape[0], dtype=np.float32)
    p1 = c1 / w1
    for i in range(indices.shape[0]):
        ind2 = indices[i]

        if scipy.sparse.issparse(c):
            c2 = (c[ind2, statesKeep].toarray()[0] + unmerged[ind2] *
                  unmerged[statesKeep] / c.shape[0])
        else:
            c2 = (c[ind2, statesKeep] + unmerged[ind2]*unmerged[statesKeep] /
                  c.shape[0])

        p2 = c2 / w[ind2]
        cp = c1 + c2
        cp /= (w1 + w[ind2])
        d[i] = c1.dot(np.log(p1/cp)) + c2.dot(np.log(p2/cp))
    return d


def absorb(c, absorb_states):
    """Absorb states into their kinetically nearest neighbors.

    Parameters
    ----------
    c : array, shape=(n_states, n_states)
        Transition counts matrix
    absorb_states : iterable
        List of states to absorb to their kinetically nearest neighbor.

    Returns
    -------
    c : array, shape=(n_states - n_absorbed, n_states - n_absorbed)
        Transition counts matrix with states absorbed
    labels : array, shape=(n_states,)
        Array of labels showing how states were absorbed.
    """

    c = c.tolil() if scipy.sparse.issparse(c) else c.copy()

    # each state starts off labeled as itself
    labels = np.arange(c.shape[0])

    for s in absorb_states:
        # first, store then zero out the self counts of the state to
        # trim so it isn't considered in argmax calculations
        self_cts = c[s, s]
        c[s, s] = 0

        if np.sum(c[s, :]) == 0:
            if self_cts:  # only self counts => disconnected
                raise exception.DataInvalid(
                    "State %s can't be absorbed into a neighbor because "
                    "it is disconnected." % s)
            else:  # the entire row is zeros => ignore
                labels[s] = -1
                continue

        if scipy.sparse.issparse(c):
            dest = c.rows[s][np.argmax(c.data[s])]
        else:
            dest = c[s, :].argmax()

        # add old transitions into the destination state
        c[dest, :] += c[s, :]
        c[:, dest] += c[:, s]
        c[dest, dest] += self_cts

        c[s, :] = c[:, s] = 0
        labels = renumberMap(labels, labels[s])
        labels[s] = labels[dest]

    return c, labels


def baysean_prune(c, n_procs=1, factor=np.log(3)):
    """Prune states less than a particular bayes' factor, lumping them
    in with their kinetically most-similar neighbor.

    Parameters
    ----------
    c : array, shape=(n_states, n_states)
        Transition counts matrix
    n_procs : int
        Width of parallelization for this operation.
    factor : float, default=ln(3)
        Bayes' factor at which to prune states.
    in_place : bool, default=False
        Compute the pruning of counts matrix C in place.

    Returns
    -------
    c : array, shape=(n_states_pruned, n_states_pruned)
        Transition counts matrix after pruning
    labels : array, shape=(n_states)
        Labels of old states in new states. The value j at position i
        indicates that state i was merged into state j.
    kept_states : array, shape=(n_states)
        Array of state indices that were retained during pruning.
    """

    if scipy.sparse.issparse(c) and not hasattr(c, '__getitem__'):
        c = c.tocsr()
    else:
        c.copy()

    # get num counts in each state (or weight)
    w = np.array(c.sum(axis=1)).flatten() + 1

    # pseudo-state (just pseudo counts)
    pseud = np.ones(c.shape[0], dtype=np.float32)
    pseud /= c.shape[0]

    indices = np.arange(c.shape[0], dtype=np.int32)
    statesKeep = np.arange(c.shape[0], dtype=np.int32)
    unmerged = np.ones(c.shape[0], dtype=np.int8)

    n_ind = len(indices)
    if n_ind > 1 and n_procs > 1:
        n_procs = min(n_ind, n_procs)
        step = n_ind // n_procs
        end_index = n_ind if n_ind % step > 3 else n_ind-step

        dlims = zip(range(0, end_index, step),
                    list(range(step, end_index, step)) + [n_ind])

        with multiprocessing.Pool(processes=n_procs) as pool:
            result = pool.map(
                functools.partial(multiDistHelper, c1=pseud, w1=1, c=c, w=w,
                                  statesKeep=statesKeep, unmerged=unmerged),
                [indices[start:stop] for start, stop in dlims])

            d = np.concatenate(result)
    else:
        d = multiDistHelper(indices, pseud, 1, c, w, statesKeep, unmerged)

    # prune states with Bayes factors less than 3:1 ratio (log(3) = 1.1)
    statesPrune = np.where(d < factor)[0]
    statesKeep = np.where(d >= factor)[0]

    c, labels = absorb(c, statesPrune)

    return c, labels, statesKeep
