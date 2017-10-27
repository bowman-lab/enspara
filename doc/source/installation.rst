Installation
============

At present, since enspara is in the early development stages, we don't have a streamlined installation process.

The following steps will get you set up.

1. Confirm Anaconda is installed on local machine and create/activate environment.

2. Clone enspara from github to local machine:

.. code-block:: bash

	git clone git@github.com:gbowman/enspara.git

3. Enter enspara from home directory and:

.. code-block:: bash

	python setup.py build_ext --inplace

If setup failed, `conda install` necessary packages and rerun setup command. 

4. Return to home directory and: 

.. code-block:: bash

	mkdir modules 
	cd modules 
	ln -s ~/enspara/enspara 
	vim ~/.bashrc

5. Add the following line to the bash script:

.. code-block:: bash

	PYTHONPATH="$PYTHONPATH:/path/to/enspara"

This completes the process of installing enspara on your local machine.
