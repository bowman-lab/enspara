# stag
Statistical Trajectory Analysis and Guidance

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
self.eq_probs_ = eq_probs(tprobs)
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
