# stag
Statistical Trajectory Analysis and Guidance

## Clustering Large Trajectory Sets

Clustering usually requires a single array, but trajectories are normally fragmented in multiple files. Our `load_as_concatenated` function will load multiple trajectories into a single numpy array. The only requirement is that each trajectory have the same number of atoms. Their topologies need not match, nor must their lengths match.

The `KHybrid` class, one of the clustering algorithms we implemented, follows the scikit-learn API.

```python
import mdtraj as md

from stag.cluster import KHybrid
from stag.util.load import load_as_concatenated

top = md.load('path/to/trj_or_topology').top

# loads a giant trajectory in parallel into a single numpy array.
lengths, xyz = load_as_concatenated(
    reversed(['path/to/trj1', 'path/to/trj2', ...]),
    top=top,
    processes=8)

# configure a KHybrid (KCenters + KMedoids) clustering object
# to use rmsd and stop creating new clusters when the maximum
# RMSD gets to 2.5A.
clustering = KHybrid(
    metric=md.rmsd,
    dist_cutoff=0.25)

# md.rmsd requires an md.Trajectory object, so wrap `xyz` in
# the topology.
clustering.fit(md.Trajectory(xyz=xyz, topology=top))

# the distances between each frame in `xyz` and the nearest cluster center
print(clustering.distances_)
# the cluster id for each frame in `xyz`
print(clustering.labels_)
# a list of the `xyz` frame index for each cluster center
print(clustering.center_indices_)
```

## Making an MSM

### Option 1: Use the object

[WIP]

### Option 2: Functional interface

```python

from stag.msm import builders
from stag.msm.transition_matrices import assigns_to_counts, TrimMapping, \
    eq_probs, trim_disconnected

lag_time = 100

tcounts = assigns_to_counts(assigns, lag_time=lag_time)

#if you want to trim states without counts in both directions:
mapping, tcounts = trim_disconnected(tcounts)

tprobs = builders.transpose(tcounts)
eq_probs_ = eq_probs(tprobs)
```

## Logging

STAG uses python's logging module. Each file has its own logger, which are
usually set to output files with the module name (e.g. `stag.cluster.khybrid`).

They can be made louder or quieter on a per-file level by accessing the
logger and running `logger.setLevel()`. So the following code sets the log
level of `util.load` to DEBUG.

```python
import logging

logging.getLogger('stag.util.load').setLevel(logging.DEBUG)
```

# Developing

## Testing

Tests are currently done with a mix of [nose](https://nose.readthedocs.io) and [unittest](https://docs.python.org/2/library/unittest.html), but new tests should be written with nose.

To run the tests, run

```
nosetests
```

You can rerun tests that failed last time with the `--failed` flag, and get extra verbose output with the `-vv` flag.
