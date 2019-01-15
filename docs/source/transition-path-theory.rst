Transition Path Theory
======================

Transition path theory is a mathematical framework to analyze MSMs to find
transition pathways.

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
