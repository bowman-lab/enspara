Transition Path Theory
======================

Transition path theory is a mathematical framework to analyze MSMs to find
transition pathways.

Computing Mean First Passage Times (MFPTs)
------------------------------------------
    Enspara implements fast, raw-matrix transition path theory (i.e. there is no dependence
    on any enspara-specific objects) for use in extracting various parameters derived in TPT.
    Among these parameters are Mean First Passage Times, which represent the mean time
    required to reach a particular state from some other specific state.

    First, you'll need a transition probability matrix and (optionally) equilibrium
    probabilities. For this recipe, we'll use the enspara `MSM` class, but any transition
    probability matrix and equilibrium probability distribution (as a numpy array) works.


.. code-block:: python

    from enspara import tpt
    from enspara.msm import MSM, builders

    msm = MSM(lag_time=10, method=builders.transpose)
    msm.fit(a)

    # mfpts is an array where mfpts[i, j] gives you the mfpt from i to j
    mfpts = tpt.mfpts(tprob=msm.tprobs_, populations=msm.eq_probs_)


Extracting maximum flux pathways
--------------------------------
    You can also extract a maximum flux pathway. First need a transition probability matrix
    and (optionally) equilibrium probabilities.

.. code-block:: python

    # assuming we've fit an MSM, as above in the MFPT example
    source_state = 1
    sink_state = 100

    # compute the net flux matrix from our msm
    nfm = tpt.net_fluxes(
        msm.tprobs_,
        source_state, sink_state,
        populations=msm.eq_probs_)

    path, flux = tpt.top_path(maximizer_ind, minimizer_ind, nfm.todense())
