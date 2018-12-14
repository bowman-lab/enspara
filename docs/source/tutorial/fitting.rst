Fitting
=======

Once you've clustered your data, you'll want to build a Markov state model.
Because this process is usually much less computationally expensive than
clustering, we'll drop into a python shell (or even better, jupyter notebook)
to do it.


.. code-block:: python

    from enspara import msm
	from enspara import ra

	assigs = ra.load('fs-khybrid-clusters0020-assignments.h5')

This uses ``enspara``'s ``RaggedArray`` submodule to load the assignments. In
this case, all your trajectories are uniform length, so you will actually get
back a ``numpy`` ``ndarray``, but in a more realistic situation when
trajectories have different lengths, this would be a ragged array.

Implied Timescales
------------------

One way of assessing the quality of your MSM is to look at the implied
timescales. In particular, this is often used to choose a lag time. Ideally,
your MSM's behavior isn't very dependent on your choice of lag time (i.e. it
satisfies the Markov assumption), and so this is usually a good thing to check.

.. code-block:: python

	import numpy as np
	from functools import partial

	# make 20 different lag times (integers) evenly spaced between 10 and 750
	lag_times = np.linspace(10, 750, num=20).astype(int)

	implied_timescales = []
	for time in lag_times:
	    m = msm.MSM(
	        lag_time=time,
	        method=msm.builders.transpose))
	    m.fit(assigs)

	    implied_timescales.append(
	        -time / np.log(msm.eigenspectrum(m.tprobs_, n_eigs=3)[0][1:3])
	    )

This will calculate the top 3 implied timescales across that range of lag times.
