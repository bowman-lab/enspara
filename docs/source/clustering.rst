Clustering
==========

Once you have your simulations run, generally the first step in building an MSM
is clustering your trajectories based on a parameter of interest. Commonly,
you'll want to look at the root mean square deviation of the states or the
euclidean distance between some sort of feature vector.

In :code:`enspara`, this functionality is availiable at three levels of detail.

1. :ref:`Apps <clustering-app>`. Clustering code is availiable in a 
command-line application that is capable of handling much of the bookkeeping
necessary for more complex clustering operations.

2. :ref:`Objects <clustering-object>`. Clustering code is wrapped into 
sklearn-style objects that offer simple API access to clustering algorithms 
and their parameters.

3. Functions. Clustering code is ultimately implemented as functions, which
offer the highest degree of control over the function's behavior, but also
require the most work on the user's part.


.. _clustering-app:

Clustering App
--------------------------------

Clustering functionality is availiable in :code:`enspara` in the script
:code:`apps/cluster.py`. Its help output explains at a granular level of detail
what it is capable of, and so here we will seek to provide a high-level
discussion of how it can be used.

When clustering, you will need to make a few important choices:

1. What type of data will you be clustering? The app accepts trajectories of
coordinates as well as arrays of vectors.

2. Which clustering algorithm will you use? We currently implement k-centers
and k-hybrid.

3. How "much" clustering will you do? Both k-centers and k-hybrid require the
choice of k-centers stopping criteria, and k-hybrid additionally requires the
choice of number of k-medoids refinements.

4. How will you compare frames to one another (i.e. what is your distance
function)? Options include RMSD (for coordinates), as well as euclidean and
manhattan distances.


A Simple Example
~~~~~~~~~~~~~~~~

One thing :code:`enspara` excels as is generating fine-grained state spaces
by clustering using RMSD as a criterion. This is very fast, and is not only
thread-parallelized to use all cores on a single computer (hat tip to MDTraj's
blazing fast RMSD calculations), but also can be parallelized across many
computers with MPI.

In a simple case, such a clustering will look something like this:

.. code-block:: bash

    python /home/username/enspara/enspara/apps/cluster.py \
      --trajectories /path/to/input/trj1.xtc /path/to/input/trj2.xtc \
      --topology /path/to/input/topology.top \
      --algorithm khybrid \
      --cluster-number 1000 \
      --distances /path/to/output/distances.h5 \
      --center-features /path/to/output/centers.pickle \
      --assignments /path/to/output/assignments.h5

This will make 1000 clusters using the k-hybrid clustering algorithm based
on all the atomic coordinates in :code:`trj1.xtc` and :code:`trj2.xtc`. Based
on the clusters it discovers, it will generate three files:

1. Centers file (:code:`centers.pickle`). This file, which is a python list of
:code:`mdtraj.Trajectory` trajectory objects, contains the atomic coordinates
that were at the center of each center. If 1000 clusters are discovered, this
list will have length 1000.

2. Assignments file (:code:`assignments.h5`). This file assigns each frame in
the input to each cluster center (even if subsampling is specified). If
:code:`(i, j)` in this array has value :code:`n`, then the :code:`j`th frame of
trajectory :code:`i` above was found to belong to center :code:`n` (found in
the centers file).

3. Distances file (:code:`distance.h5`). This file gives the distance between
each frame :code:`(i, j)` and the center it is assigned to (found in the
assignments file).

Atom Selection and Shared State Spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to cluster proteins with differing topologies into the same
state space. To do this, we rely on the :code:`--atoms` flag to select matching
atoms between the two topologies. The :code:`--atoms` flag uses the
`MDTraj DSL <http://mdtraj.org/latest/atom_selection.html>`_
selection syntax to specify which atoms will be loaded from each trajectory.

Imagine we have simulations of a wild-type and point mutant. To specify the
the different trajectories and topologies, we pass :code:`--trajectories` and
:code:`--topology` more than once. Then, we pass :code:`--atoms` to indicate
which atoms should be taken. In this example, we will take just the alpha
carbons.

.. code-block:: bash

    python /home/username/enspara/enspara/apps/cluster.py \
      --trajectories wt1.xtc wt2.xtc \
      --topology wt.top \
      --trajectories mut1.xtc mut.xtc \
      --topology mut.top \
      --atoms 'name CA' \
      --algorithm khybrid \
      --cluster-number 1000 \
      --distances /path/to/output/distances.h5 \
      --center-features /path/to/output/centers.pickle \
      --assignments /path/to/output/assignments.h5

Feature Clustering
~~~~~~~~~~~~~~~~~~

Enspara can also operate on inputs that are "features" rather than coordinates.
For example, we have published work that uses clusters based on the solvent
accessibility of each sidechain, rather than their position. In that
featurization each frame is represented by a one-dimensional vector, and the
distances between vectors is computed using some distance function, often
the euclidean or manhattan distance (both of which have fast implementations in
:code`enspara`).

In this case, your :code:`cluster.py` invocation will look something like:

.. code-block:: bash

    python /home/username/enspara/enspara/apps/cluster.py \
      --features features.h5 \
      --algorithm khybrid \
      --cluster-radius 1.0 \
      --cluster-distance euclidean \
      --distances /path/to/output/distances.h5 \
      --centers /path/to/output/centers.pickle \
      --assignments /path/to/output/assignments.h5

Here, clusters will be generated until the maximum distance of any frame to its
cluster center is 1.0 using a Euclidean distance (the :code:`--cluster-number`
flag is also accepted). You can also specify a list of npy files 

Subsampling and Reassignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, it is useful not to load every frame of your trajectories. This can
be necessary for large datasets, where the data exceeds the memory capacity of
the computer(s) being used for clustering, and often does not substantially
diminish the quality of the clustering. As a general rule of thumb, it is
usually safe to subsample such that frames are 1 ns apart. Thus, if frames have
been saved every 10 ps, subsampling by a factor 100 is usually safe. This can
be achieved with the :code:`--subsample` flag.

.. code-block:: bash

    python /home/username/enspara/enspara/apps/cluster.py \
      --trajectories /path/to/input/trj1.xtc /path/to/input/trj2.xtc \
      --topology /path/to/input/topology.top \
      --algorithm khybrid \
      --subsample 10 \
      --cluster-number 1000 \
      --distances /path/to/output/distances.h5 \
      --center-features /path/to/output/centers.pickle \
      --assignments /path/to/output/assignments.h5

However, when clustering is produced with a subset of the data, it is still
valuable to use all frames to build a Markov state model, because it improves
the statistics in the transition counts matrix. Consequently, even when
clustering uses some subset of frames, it is useful to assign every frame in
the dataset to a cluster. This process is called "reassignment".

By default, reassignment automatically occurs after clustering (it can be
suppressed with :code:`--no-reassign`). It sequentially loads subsets of the
input data (the size of the subset depends on the size of main memory) and
uses the cluster centers to determine cluster membership before purging the
subset from memory and loading the next.

Notably, reassignment is embarassingly parallel, whereas clustering is
fundamentally less scalable. As a result, one can provide the
:code:`--no-reassign` flag to suppress this behavior and use the centers in
some other script to do the reassignment (see the :code:`reassign.py` app).

.. _clustering-object:

Clustering Object
-----------------

Rather than relying on a pre-built script to cluster data, there is also a
scikit-learn-like object for the two major clustering algorithms we use,
k-hybrid and k-centers. They are :any:`enspara.cluster.hybrid.KHybrid` and
:any:`enspara.cluster.kcenters.KCenters`, respectively.

An example of a script that clusters data using this object is:

.. code-block:: python

    import mdtraj as md

    from enspara.cluster import KHybrid
    from enspara.util.load import load_as_concatenated

    top = md.load('path/to/trj_or_topology').top

    # loads a giant trajectory in parallel into a single numpy array.
    lengths, xyz = load_as_concatenated(
        ['path/to/trj1', 'path/to/trj2', ...],
        top=top,
        processes=8)

    # configure a KHybrid (KCenters + KMedoids) clustering object
    # to use rmsd and stop creating new clusters when the maximum
    # RMSD gets to 2.5A.
    clustering = KHybrid(
        metric=md.rmsd,
        dist_cutoff=0.25)

    # md.rmsd requires an md.Trajectory object, so wrap `xyz` in
    # the topology.
    clustering.fit(md.Trajectory(xyz=xyz, topology=top))

    # the distances between each frame in `xyz` and the nearest cluster center
    print(clustering.distances_)

    # the cluster id for each frame in `xyz`
    print(clustering.labels_)

    # a list of the `xyz` frame index for each cluster center
    print(clustering.center_indices_)


.. _clustering-function:

Clustering Functions
--------------------

Finally,  for the finest-grained control over the clustering process, we implement
functions that execute the clustering algorithm over given data, often with very
detailed control over stopping conditions and calculations. They are 
:any:`enspara.cluster.hybrid.hybrid` and :any:`enspara.cluster.kcenters.kcenters`, respectively.
