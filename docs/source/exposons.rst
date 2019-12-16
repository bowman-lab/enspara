Exposons
========

Exposons [EXP2019]_ are a method for identifying cooperative changes at a protein's surface. They have been shown to identify cryptic pockets (potentially-druggable concavities on a protein's surface that are not easily identified by traditional structural techniques), as well as other types of allosteric rearrangement at a protein's surface.

We have included a method for computing exposons on an MSM in enspara, although some of the computational costs of doing so may require additional work to make scalable to large systems.

Quick and Dirty Exposons
------------------------

For small systems, exposons can be quickly calculated with the :any:`enspara.info_theory.exposons.exposons` method:


.. code-block:: python

    mi, exposons = exposons(trj, damping=0.9, weights=eq_probs)

.. warning::

    In :any:`enspara.info_theory.exposons.exposons_from_sasas`, assumptions are made about the name of your atoms. See `A Note on Atoms' Names`_ for details.

The key decisions here are which trajectory to use (typically you'll want to use a representative conformation for each state in a sufficiently fine-grained MSM), the damping parameter (for :any:`sklearn.cluster.AffinityPropagation`), and the `weights` parameter. In the case of an MSM, the `weights` are the equilibrium probabilities for each state. Otherwise, it can be omitted, and is just assumed to be :math:`1/n`, where :math:`n` is the length of `trj`.

The output is a 2-tuple of an MI matrix, which gives the mutual information between the solvent accessibility state of each pair of residues :math:`i` and :math:`j`, and a numpy array, which gives the assignment of each residue to an exposon.


Exposons for Larger Systems
---------------------------

If you find that exposons take too long to calculate with the convenience wrapper above, you will likely need to split the calcuation up into multiple parts. Early stages of exposons calcuation (such as SASA calculations) are embarassingly parallel in the number of trajectory frames.

Exposons are calculated in three primary phases:

1. Atomic solvent acessible surface area (SASA) calculation, which relies entirely on MDTraj's implementation of the Shrake-Rupley algorithm:

.. code-block:: python

    sasas = md.shrake_rupley(trj, probe_radius=probe_radius, mode='atom')

2. Condensation of the atomic SASA into sidechain SASA:

.. code-block:: python

    from enspara.info_theory.exposons import condense_sidechain_sasas

    sasas = condense_sidechain_sasas(sasas, trj.top)

3. Calculation of exposons from sasa-featuized data:

.. code-block:: python

    from enspara.info_theory.exposons import exposons_from_sasas

    exposons = exposons_from_sasas(sasas, damping, weights, threshold)


A Note on Atoms' Names
----------------------

As you are likely aware, there are numerous schemes for naming atoms in protein topologies. The code in :any:`enspara.info_theory.exposons.condense_sidechain_sasas` code does not do anything sophisticated with respect to this and, indeed, is only aware of the GROMACS naming scheme. Specifically, sidechains are classified as any resdue matching the following query:

.. code-block::

  not (name N or name C or name CA or name O or name HA or
       name H or name H1 or name H2 or name H3 or name OXT)

Because different softwares name their atoms differently, there are no guarantees whatsoever that this matches for your protein. Please be aware of this. Users interested in improving to the intelligence of this code are encouraged to propose (or better submit) solutions on `GitHub <https://github.com/bowman-lab/enspara>`_.

.. [EXP2019] Justin R Porter, Katelyn E Moeder, Carrie A Sibbald, Maxwell I Zimmerman, Kathryn M Hart, Michael J Greenberg, and Gregory R Bowman. "Cooperative Changes in Solvent Exposure Identify Cryptic Pockets, Switches, and Allosteric Coupling." Biophysical Journal, January 2019. https://doi.org/10.1016/j.bpj.2018.11.3144.
