[![DOI:10.1007/978-3-319-76207-4_15](https://zenodo.org/badge/DOI/10.1063/1.5063794.svg)]( https://doi.org/10.1063/1.5063794)

[![Build Status](https://github.com/bowman-lab/enspara/actions/workflows/ci.yml/badge.svg)](https://github.com/bowman-lab/enspara/actions/)


# enspara
MSMs at Scale 

## Reference

If you use `enspara` for published research, please cite us:

Porter, J.R., Zimmerman, M.I. and Bowman, G.R., 2019. [Enspara: Modeling molecular ensembles with scalable data structures and parallel computing.](https://aip.scitation.org/doi/full/10.1063/1.5063794%40jcp.2019.MMMK.issue-1) The Journal of chemical physics, 150(4), p.044108.

## Installation 

Installation is documented [here](https://enspara.readthedocs.io/en/latest/installation.html).
```
conda create -n enspara
conda install -c bowmanlab -c conda-forge enspara
```

Alternatively if you wish to build the latest:

```
git clone https://github.com/bowman-lab/enspara
mamba create -n enspara -c conda-forge cython numpy mdtraj scipy python=3.12 mpi4py
mamba activate enspara
cd enspara
pip install -e .
```

Optionally, install pytests to run the tests:
`mamba install -c conda-forge pytest`

## Building the docs

Enspara uses sphinx for documentation. They're a bit of a work in progress,
but most of the most important stuff is documented already.

```bash
cd docs
make html
```

## Running the tests

Enspara uses `pytest` as a test discovery and running tool. To run the
tests, you should first make sure you have the development dependencies
installed then, from the enspara directory, run:

```bash
pytest
```

By default this runs without MPI. 

If you wish to explicitly skip the MPI tests, you can run:

```bash
pytest -m 'not mpi'
```

If you then want to run the mpi tests (including the MPI ones), you can additionally run:

```bash
mpirun -n 2 python -m pytest -m 'mpi'
```
