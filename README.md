# stag
Statistical Trajectory Analysis and Guidance

# Developing

## Testing

Tests are currently done with a mix of [nose](https://nose.readthedocs.io) and [unittest](https://docs.python.org/2/library/unittest.html), but new tests should be written with nose.

To run the tests, run

```
nosetests
```

You can rerun tests that failed last time with the `--failed` flag, and get extra verbose output with the `-vv` flag.