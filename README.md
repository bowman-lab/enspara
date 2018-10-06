[![Circle CI](https://circleci.com/gh/bowman-lab/enspara.svg?style=svg)](https://circleci.com/gh/bowman-lab/enspara)

# enspara
MSMs at Scale 

## Installation 
1) Confirm Anaconda is installed on local machine and create/activate environment.
2) Clone enspara from github to local machine: \
`git clone git@github.com:gbowman/enspara.git` 
3) Enter enspara from home directory and: \
`python setup.py build_ext --inplace` 
If setup failed, `conda install` necessary packages and rerun setup command. 
4) Return to home directory and: 
```
mkdir modules 
cd modules 
ln -s ~/enspara/enspara 
vim ~/.bashrc
```
5) Add the following line to the bash script: \
`PYTHONPATH="directory path to modules:$PYTHONPATH"`

This completes the process of installing enspara on your local machine.

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
