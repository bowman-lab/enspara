import logging
import functools
import multiprocessing

import numpy as np

import scipy
import scipy.io
import scipy.sparse

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


def bace(c, nMacro, nProc, prune_fn, chunkSize=100):
    # perform filter
    logger.info("Checking for states with insufficient statistics")
    c, state_map, statesKeep = prune_fn(c, nProc)
    c = c.astype('float')

    # get num counts in each state (or weight)
    w = np.array(c.sum(axis=1)).flatten()
    w[statesKeep] += 1

    unmerged = np.zeros(w.shape[0], dtype=np.int8)
    unmerged[statesKeep] = 1

    # get nonzero indices in upper triangle
    indRecalc = getInds(c, statesKeep, chunkSize)
    if scipy.sparse.issparse(c):
        dMat = scipy.sparse.lil_matrix(c.shape)
    else:
        dMat = np.zeros(c.shape, dtype=np.float32)

    if scipy.sparse.issparse(c):
        c = c.tocsr()

    bayes_factors = {}
    state_maps = []

    dMat, minX, minY = calcDMat(c, w, bayes_factors, indRecalc, dMat, nProc,
                                statesKeep, unmerged, chunkSize)
    logger.info("Coarse-graining...")

    for cycle in range(c.shape[0] - nMacro):
        logger.info("Iteration %d, merging %d states",
                    cycle, c.shape[0] - cycle)
        rslt = mergeTwoClosestStates(
            c, w, bayes_factors, indRecalc, dMat, nProc, state_map,
            statesKeep, minX, minY, unmerged, chunkSize)
        c, w, indRecalc, dMat, state_map, statesKeep, unmerged, minX, \
            minY = rslt

        state_maps.append(state_map.astype(int))

    state_maps.append(np.zeros((c.shape[0])))

    return bayes_factors, np.vstack(state_maps[::-1])


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


def calcDMat(c, w, bayes_factors, indRecalc, dMat, nProc, statesKeep,
             unmerged, chunkSize):
    nRecalc = len(indRecalc)
    if nRecalc > 1 and nProc > 1:
        if nRecalc < nProc:
            nProc = nRecalc
        pool = multiprocessing.Pool(processes=nProc)
        n = len(indRecalc)
        stepSize = int(n/nProc)
        if n % stepSize > 3:
            dlims = zip(
                range(0, n, stepSize),
                list(range(stepSize, n, stepSize)) + [n])
        else:
            dlims = zip(
                range(0, n-stepSize, stepSize),
                list(range(stepSize, n-stepSize, stepSize)) + [n])
        args = []
        for start, stop in dlims:
            args.append(indRecalc[start:stop])
        result = pool.map_async(
            functools.partial(multiDist, c=c, w=w, statesKeep=statesKeep,
                              unmerged=unmerged, chunkSize=chunkSize), args)
        result.wait()
        d = np.vstack(result.get())
        pool.close()
    else:
        d = multiDist(indRecalc, c, w, statesKeep, unmerged, chunkSize)
    for i in range(len(indRecalc)):
        dMat[indRecalc[i][0], indRecalc[i][1]] = d[i][:len(indRecalc[i][1])]

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


def baysean_prune(c, n_procs=1):
    '''Prune states less than a particular bayes' factor, lumping them
    in with their kinetically most-similar neighbor.

    Parameters
    ----------
    c : array, shape=(n_state, n_states)
        Transition counts matrix
    n_procs: int
        Width of parallelization for this operation.
    '''

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

        if n_ind % step > 3:
            dlims = zip(
                range(0, n_ind, step),
                list(range(step, n_ind, step)) + [n_ind])
        else:
            dlims = zip(
                range(0, n_ind-step, step),
                list(range(step, n_ind-step, step)) + [n_ind])

        with multiprocessing.Pool(processes=n_procs) as pool:
            result = pool.map_async(
                functools.partial(multiDistHelper, c1=pseud, w1=1, c=c, w=w,
                                  statesKeep=statesKeep, unmerged=unmerged),
                [indices[start:stop] for start, stop in dlims])

            result.wait()
            d = np.concatenate(result.get())
    else:
        d = multiDistHelper(indices, pseud, 1, c, w, statesKeep, unmerged)

    # prune states with Bayes factors less than 3:1 ratio (log(3) = 1.1)
    statesPrune = np.where(d < 1.1)[0]
    statesKeep = np.where(d >= 1.1)[0]
    logger.info("Merging %d states with insufficient statistics into their "
                "kinetically-nearest neighbor", statesPrune.shape[0])

    # init map from micro to macro states
    state_map = np.arange(c.shape[0], dtype=np.int32)

    for s in statesPrune:
        dest = c[s, :].argmax()
        c[dest, :] += c[s, :]
        c[s, :] = 0
        c[:, s] = 0
        state_map = renumberMap(state_map, state_map[s])
        state_map[s] = state_map[dest]

    return c, state_map, statesKeep
