Installation
============

Enspara can be installed from our github repository in the following way:

1. Confirm Anaconda is installed on local machine and create/activate environment.

2. Clone enspara from github to local machine:

.. code-block:: bash

	git clone https://github.com/bowman-lab/enspara

3. Install compiled dependecies of enspara (one frequently runs into problems compiling these locally via `pip`, but if you prefer you can just run `setup.py` and `pip` will try to download an compile these packages):

.. code-block:: bash

        conda install -c conda-forge mdtraj=1.8.0
	conda install numpy==1.14
	conda install cython
	conda install mpi4py -c conda-forge

4. Enter enspara from home directory and:

.. code-block:: bash

	python setup.py install

If setup failed, `conda install` necessary packages and rerun setup command. 

5. Check that you've been successful:

.. code-block:: bash

	cd && python -c 'import enspara'
