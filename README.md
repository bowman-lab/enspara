[![Circle CI](https://circleci.com/gh/bowman-lab/enspara.svg?style=svg)](https://circleci.com/gh/bowman-lab/enspara)

# enspara
MSMs at Scale 

## Installation 

Installation is documented [here](https://enspara.readthedocs.io/en/latest/installation.html).

## Building the docs

Enspara uses sphinx for documentation. They're a bit of a work in progress,
but most of the most important stuff is documented already.

```bash
cd docs
make html
```

## Running the tests

Enspara uses `nosetests` as a test discovery and running tool. To run the
tests, you should first make sure you have the development dependencies
installed then, from the enspara directory, run:

```bash
nosetests enspara
```

If you want to run the MPI tests, you can run

```bash
mpiexec -n 2 nosetests enspara -a mpi
```

where `-a mpi` asks nose to run only the tests that are MPI tests.
