# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numpy as np

def energy_to_probability(u, kT=2.479):
    p = np.exp(-(u-u.mean())/kT)
    p /= p.sum()
    return p

def shannon_entropy(p):
    inds = np.where(p>0)[0]
    H = -p[inds].dot(np.log(p[inds]))
    return H

def relative_entropy(p, q):
    p_inds = np.where(p>0)[0]
    q_inds = np.where(q>0)[0]
    inds = np.intersect1d(p_inds, q_inds)
    rel_ent = p[inds].dot(np.log(p[inds]/q[inds]))
    return rel_ent
    
def js_divergence(p, q):
    m = 0.5*(p+q)
    js = 0.5*relative_entropy(p,m) + 0.5*relative_entropy(q,m)
    return js
    