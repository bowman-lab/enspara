Installation
============

Enspara can be installed from our github repository in the following way:

1. Confirm Anaconda is installed on local machine and create/activate environment.

2. Clone enspara from github to local machine:

.. code-block:: bash

	git clone git@github.com:gbowman/enspara.git

3. Install the dependencies of ``setup.py`` (``setuptools`` makes it hard to streamline this):

.. code-block:: bash

	pip install numpy cython

4. Enter enspara from home directory and:

.. code-block:: bash

	python setup.py install

If setup failed, `conda install` necessary packages and rerun setup command. 

5. Check that you've been successful:

.. code-block:: bash

	cd && python -c 'import enspara'
