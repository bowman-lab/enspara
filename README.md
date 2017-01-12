# stag
Statistical Trajectory Analysis and Guidance

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
