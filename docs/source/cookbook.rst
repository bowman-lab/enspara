Cookbook
========

This group of handy recipies might be helpful if you're looking to do something pretty specific and pretty common.

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


Coarse-graining with BACE
-------------------------

.. danger::
    Be warned that our BACE implementaiton is still experimental, and you should be
    careful to check your output.

BACE is an algorithm for converting a large, fine-grained Markov state model into
a smaller, coarser-grained model.

.. code-block:: python

    from enspara import array as ra
    from enspara import msm

    assigs = ra.load('path/to/assignments.h5')

    m = msm.MSM(lag_time=20, method=msm.builders.transpose)
    m.fit(assigs)

    bayes_factors, labels = msm.bace.bace(m.tcounts_, n_macrostates=2, n_procs=8)

This code will create two dictionaries, ``bayes_factors``, which contains a mapping from
number of microstates (up to ``n_microstates`` as specified in the call to ``bace()``) to
a the Bayes' factor for the model with that number of microstates, and ``labels``, a
mapping from number of microstates to a labeling of the initial microstates of ``m`` into
a that number of microstates.


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
