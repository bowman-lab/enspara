Cookbook
========

This group of handy recipies might be helpful if you're looking to do something pretty specific and pretty common.


Clustering Large Trajectory Sets
--------------------------------

Something enspara really excels as it handling gigantic tests of data. Enspara has been used to build MSMs on of sets of trajectories with sizes in the TB range.

Clustering usually requires a single array, but trajectories are normally fragmented in multiple files. Our `load_as_concatenated` function will load multiple trajectories into a single numpy array. The only requirement is that each trajectory have the same number of atoms. Their topologies need not match, nor must their lengths match.

The `KHybrid` class, one of the clustering algorithms we implemented, follows the scikit-learn API.

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


Building an MSM
---------------

Using the object-level interface

.. code-block:: python

    from enspara.msm import MSM, builders

    # build the MSM fitter with a lag time of 100 (frames) and
    # using the transpose method
    msm = MSM(lag_time=100, method=builders.transpose)

    # fit the MSM to your assignments (a numpy ndarray or ragged array)
    msm.fit(assignments)

    print(msm.tcounts_)
    print(msm.tprobs_)
    print(msm.eq_probs_)

Using the function-level interface

.. code-block:: python

    from enspara.msm import builders
    from enspara.msm.transition_matrices import assigns_to_counts, TrimMapping, \
        eq_probs, trim_disconnected

    lag_time = 100

    tcounts = assigns_to_counts(assigns, lag_time=lag_time)

    #if you want to trim states without counts in both directions:
    mapping, tcounts = trim_disconnected(tcounts)

    tprobs = builders.transpose(tcounts)
    eq_probs_ = eq_probs(tprobs)


Changing logging
----------------
    Enspara uses python's logging module. Each file has its own logger, which are
    usually set to output files with the module name (e.g. `enspara.cluster.khybrid`).

    They can be made louder or quieter on a per-file level by accessing the
    logger and running `logger.setLevel()`. So the following code sets the log
    level of `util.load` to DEBUG.

.. code-block:: python

    import logging

    logging.getLogger('enspara.util.load').setLevel(logging.DEBUG)


Extracting maximum flux pathways
--------------------------------
    Enspara implements fast, raw-matrix transition path theory (i.e. there is no dependence
    on any enspara-specific objects) for use in extracting various parameters derived in TPT.
    This includes extracting maximum flux pathways.

    To extract a maximum flux pathway, you first need a transition probability matrices and
    (optionally) equilibrium probabilities. For the purposes of this recipe, we'll use the
    enspara `MSM` class, but any transition probability matrix and equilibrium probability
    distribution will work!

.. code-block:: python

    import msmbuilder.tpt

    from enspara import tpt
    from enspara.msm import MSM, builders

    msm = MSM(lag_time=10, method=builders.transpose)
    msm.fit(assignments)

    source_state = 1
    sink_state = 100

    # compute the net flux matrix from our 
    nfm = tpt.net_fluxes(
        msm.tprobs_,
        source_state, sink_state,
        populations=msm.eq_probs_)

    path, flux = msmbuilder.tpt.top_path(maximizer_ind, minimizer_ind, nfm.todense())
