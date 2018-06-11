Apps
========

These apps make it easy to do the common tasks associated with building an MSM.


RMSD Clustering
--------------------------------

Once you have your simulations run, the first thing you'll want to do is 
cluster your trajectories based on a parameter of interest. Commonly, you'll 
want to look at the root mean square deviation of the states. You can do this 
using rmsd_cluster.py

The app requires trajectory files and their corresponding topology file(s).
All file types that MDTraj supports are supported here.

.. code-block:: bash

    traj=/path/to/directory/with/trajectories/*.xtc
    top=/path/to/directory/with/topology/file_name.pdb
    
Probably the least trivial selection is the residues to cluster based upon.
You must use MDTraj atom-selection syntax. Below an example.

.. code-block:: bash

    selection1='(name C or name O or name CA or name N or name CB)'
    # This selects certain atom types
    selection2='((residue 40 to 43) or (residue 237 to 243) or (residue 266 to 272))'
    # This selects certain residues of interest
    selection3='((name CB) and (residue 42 or residue 238))'
    # This de-selects certain atoms
    # This is useful when creating a shared state space between mutants
    selection="$selection and $selection2 and not $selection3"

Note: If you have multiple proteins for which you wish to create a shared state
space, you must supply a topology file and atom selection for each trajectory set.
These must be described directly below the corresponding trajectories.

.. code-block:: bash

    --trajectories traj_wt
    --topology top_wt
    --atoms atoms_wt
    
    --trajectories traj_mutant
    --topology top_mutant
    --atoms atoms_mutant
    

The app also requires an algorithm to use for clustering. Currently, the 
options are "khybrid" or "kcenters".

There are a variety of parameters you can also set.

.. code-block:: bash

    --rmsd-cuttoff float
    # This will determine the cluster size (radius). The units are in nanometers (nm). 
    --n-clusters integer
    # This will set the number of clusters. This will be the minimum number of clusters 
    # if you also supply a cluster radius.
    --processes integer
    # This will set the number of cores to use
    --subsample integer
    # This will take only every nth frame when loading trajectories.
    --no-reassign truth value
    # This is avaible because it can be done sepatarely when working with large data sets. 
    # The default is to reassign.

  
The outputs are files containing information on assignments, centers, and 
distances. Assignments describes which cluster each frame belongs to. Centers
describes the frame that corresponds to each cluster center. Distances tells
how far each frame is from the cluster center.

Your final submit script should be formatted something like this.

.. code-block:: bash

    python /home/username/enspara/enspara/apps/rmsd_cluster.py \
     --trajectories $traj \
     --topology $top \
     --atoms "$selection" \
     --algorithm khybrid \
     --n-clusters 1000 \
     --processes 24 \
     --subsample 100 \
     --distances /path/to/directory/distances.h5 \
     --centers /path/to/directory/centers.pickle \
     --assignments /path/to/directory/assignments.h5
     
     

Implied Timescales Plot
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
    

