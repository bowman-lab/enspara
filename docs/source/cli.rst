CLI
========

These apps make it easy to do the common tasks associated with building an MSM.


Clustering
--------------------------------

Once you have your simulations run, the first thing you'll want to do is 
cluster your trajectories based on a parameter of interest. Commonly, you'll 
want to look at the root mean square deviation of the states or the euclidean
distance between some sort of feature vector. You can do this using
:code:`apps/cluster.py`

This app is documented in :ref:`clustering-app`.

Implied Timescales
--------------------------------

Once you've clustered, you might want to know what lag time is appropiate to use
to create your MSM. You can can plot eigenvalue motion spped as a function of
lag time by using  implied_timescales.py

The app only requires the assignment files.

.. code-block:: bash

    --assignments path/to/directory/with/file.h5
    # This is the file containing assignments
    
However, there are many other parameters that can be set as well.

.. code-block:: bash

    --n-eigenvalues integer
    # This is the number of eigenvalues that will be computed for each lag time.
    # The default is five.
    --lag-times min:max:step
    # This is the list of lag times (in frames).
    # The default is 5:100:2.
    --symmetrization method name
    # This is the method to use to enforce detailed balance in the counts matrix.
    # The default is transpose.
    --trj-ids trajs
    # This will only use given trajectories to compute the implied timescales.
    # This is useful for handling assignments for shared state space clusterings.
    # The deafult is none.
    --processes integer
    # This will set the number of cores to use.
    # Because eigenvector decompositions are thread-parallelized, this should
    # usually be several times smaller than the number of cores availiable on 
    # your machine.
    # The deafult is max(1, cpu_count()/4).
    --trim truth statement
    # This will turn on ergodic trimming
    # The default is False.
    --plot path/to/directory/file_name.png
    # This is how the plot will save.
    --logscale
    # This will put the y-axis of the plot on a log scale.


Your final submit script should be formatted something like this.

.. code-block:: bash

    python /home/username/enspara/enspara/apps/implied_timescales.py \
    --assignments assignments.h5 \
    --n-eigenvalues 5 \
    --processes 2 \
    --plot implied_timescales.png \
    --logscale
    

