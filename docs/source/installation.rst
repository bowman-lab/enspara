Installation
============

Enspara can be installed from our github repository in the following way:

1. Create a pip/anaconda environment for enspara. For anaconda,

.. code-block:: bash

	conda create --name enspara

or with pip,

.. code-block:: bash

	python3 -m pip install --user virtualenv
	python3 -m venv enspara
	source enspara/bin/activate

2. Install enspara's build-time dependencies:

.. code-block:: bash

	pip install mdtraj cython

or, if you prefer anaconda,

.. code-block:: bash

	conda install mdtraj cython

3. Use pip to clone and install enspara:

.. code-block:: bash

	pip install git+https://github.com/bowman-lab/enspara

4. If you need MPI support, you can pip install mpi4py as well:

.. code-block:: bash

	pip install mpi4py


Developing
----------

To install enspara for development

1. Set up a virtual/anaconda environment, for example,

.. code-block:: bash

	python3 -m pip install --user virtualenv
	python3 -m venv enspara
	source enspara/bin/activate

2. Clone the git repository,

.. code-block:: bash

	git clone https://github.com/bowman-lab/enspara

3. Install build-time dependecies,

.. code-block:: bash

	pip install mdtraj cython

4. Build and install enspara in development mode

.. code-block:: bash

	cd enspara && pip install -e .[dev]

