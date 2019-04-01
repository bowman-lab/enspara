[![DOI:10.1007/978-3-319-76207-4_15](https://zenodo.org/badge/DOI/10.1063/1.5063794@jcp.2019.MMMK.issue-1.svg)]( https://doi.org/10.1063/1.5063794@jcp.2019.MMMK.issue-1) [![Circle CI](https://circleci.com/gh/bowman-lab/enspara.svg?style=svg)](https://circleci.com/gh/bowman-lab/enspara)

# enspara
MSMs at Scale 

## Reference

If you use `enspara` for published research, please cite us:

Porter, J.R., Zimmerman, M.I. and Bowman, G.R., 2019. [Enspara: Modeling molecular ensembles with scalable data structures and parallel computing.](https://aip.scitation.org/doi/full/10.1063/1.5063794%40jcp.2019.MMMK.issue-1) The Journal of chemical physics, 150(4), p.044108.

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
