CARDS
==========

One of the interesting phenomena that simulations can report on is allosteric 
communication. Identifying which residues communicate with a target site of interest, 
like an active site, can help isolate the important regions within a protein. 

``CARDS`` is a way of capturing this long-range communication from MD simulations, 
measuring coupling between between every pair of dihedrals in an entire protein. 
The CARDS methodology has been published in [CARDS2017]_.

Using CARDS is a two-step process: 

    1. Using the implementation within ``enspara`` to :ref:`collect cards <collect-cards>`
    by utilizing the command-line script. 

    2. Analyzing CARDS data using the CARDS-Reader library (URL) as described in the
    :ref:`analysis <analyze-cards>` section


.. _collect-cards:

Measure correlations with CARDS
--------------------------------------
CARDS works by decomposing each rotamer into two sets of states - rotameric 
states and dynamical states.  Dynamical states are determined by whether a 
dihedral remains in a single rotameric state (ordered) or rapidly transitions 
between multiple rotameric states (disordered). It then computes pairwise 
correlations between every pair of dihedrals and their rotameric and dynamical
states. Thus there are four types of correlations produced by CARDS.

    1. Structure-structure (between rotameric states)
    2. Disorder-disorder (between dynamical states)
    3. Structure-disorder (rotameric states of one dihedral with dynamical states of another)
    4. Disorder-structure (dynamical states of one dihedral with rotameric states of another)

``enspara`` features a multi-processing implementation of CARDS that is useful 
for large systems or datsets. Before you get started you will need to have the 
following: 

    1. A directory containing all your trajectory files, any format recognized by ``MDTraj`` is fine
    2. A topology file that corresponds to your trajectory files (if it doesn't have
       the topology pre-written like in `.h5` files). Again, any format recognizable
       by ``MDTraj`` is acceptable.

Once you have these two inputs, you must consider how big of a "buffer-zone" you want to use. 
To prevent counting spurious transitions as part of the correlated motions of your system, 
CARDS places a *buffer-zone* around each rotameric-barrier, defining "core-regions" within
each rotamer. Thus, a rotamer has only had a "true" transition if it enters a new "core region" 
than the one it previously occupied. Generally we use buffer-zones that are ~15 degrees 
on each side of the barrier. 

You can read more about buffer zones in the publication [CARDS2017]_.

Running CARDS 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's run a simple example of the ``collect_cards.py`` script. In our case we have some
directory containing a series of ``.xtc`` files that are MD Trajectories. 
We can catch them all using the wild-card ``*.xtc``. 
We also have our ``topology.top`` file to load in alongside the trajectories. 

.. code-block:: bash

    python /home/username/enspara/enspara/apps/collect_cards.py \
      --trajectories /path/to/input/*.xtc \
      --topology /path/to/input/topology.top \
      --buffer-size 15 \
      --processes 1 \
      --matrices /home/username/output_path/cards.pickle \
      --indices /home/username/output_path/inds.csv \

This is a CARDS run that will use a buffer-width of 15 degrees, and a single core. 
The outputs are 1) a single pickle file containing a dictionary of the four types of matrices
that CARDS generates, and 2) An `inds.csv` file that contains the atom indices that
correspond to each row/column in the CARDS matrices. 

Specifically, the ``cards.pickle`` output is a dictionary that contains four matrices. The 
dictionary keys identify which matrix measures which type of correlation,


.. _analyze-cards:

Analyze CARDS data
--------------------------------
Analysis of CARDS data can be done using the `CARDS-Reader library <https://github.com/sukritsingh/cardsReader>`_
As published in CARDS-using papers, there are multiple ways CARDS data can be
analyzed, including: 

    1. Extracting Shannon entropies to measure residue disorder  
    2. Computing a holistic correlations matrix. 
    3. Target site analysis 



Extracting Shannon entropy to measure residue disorder 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Shannon entropy is an information-theoretic metric that can be used to measure
disorder in a dataset. It is computed across a dataset by looking at the population of each bin: 

:math:`H(X) = -\Sigma_{x} P(x)log(P(x))`

In CARDS, it provides a useful insight into how much any single dihedral moves around 
across the simulation dataset. 

Computing Shannon entropy of how a dihedral moves in inherently a structural phenomena, 
and conveniently is equivalent to the diagonal of the CARDS matrix corresponding to 
Structure_Structure correlation (between rotameric states). We also want to be able to understand
motion on an amino-acid level, rather than a dihedral level. 

In CARDS-Reader we can use the ``apps/extract_dihedral_entropy.py`` found inside the 
CARDS-Reader library. 

.. code-block:: bash

    python /home/username/cardsReader/apps/extract_dihedral_entropy.py \
      --matrices /home/username/output_path/cards.pickle \
      --indices /home/username/output_path/inds.csv \
      --topology /path/to/input/topology.top \

In this script you are simply inputing the same topology file as used in `collect_cards.py` 
and the outputs from ``collect_cards.py``. 

The output will be two files, ``dihedral_entropy.csv`` and ``residue_entropy.csv`` that will 
have the entropy for each dihedral (AKA just the diagonal), and the residue-level entropy,
which is normalized by the maximum amount of entropy a residue can have. In other words, 
a residue-level entropy of 0.3 means a residue has ~30% of the maximum possible Shannon entropy 
value it can have. 


Computing holistic correlations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To capture the full pattern of communication into a single dihedral matrix, we can sum 
the four matrices in ``cards.pickle`` directly into a single *Holistic communication matrix*. 

This is a relatively trivial task, but for convenience, CARDS-Reader has an apps script 
``apps/generate_holistic_matrix.py`` that computes this matrix and saves it. 

.. code-block:: bash

    python /home/username/cardsReader/apps/generate_holistic_matrix.py \
      --directory /home/username/output_path/cards.pickle \


At it's core, CARDS is built on the fundamental idea that the overall communication pattern 
of a system is based on the combined communication of rotameric and disordered states.

:math:`I_{Holistic} = I_{Structural} + I_{Disordered}`

The ``generate_holistic_matrix.py`` script computes both the Structural-Structural matrix (``Structural_MI.csv``), 
as well as a single disorder-disorder matrix (``totalDisorder_MI.csv``), which is the sum of the other three matrices.
It also outputs the total Holistic communication matrix (``holistic_MI.csv``) 

This holistic communication matrix is what we can use to probe overall communication patterns in our system, 
using techniques like *Target site analysis*, or other methods.  


.. [CARDS2017] Singh, Sukrit, and Gregory R Bowman. “Quantifying Allosteric Communication via Both Concerted Structural Changes and Conformational Disorder with CARDS.” Journal of Chemical Theory and Computation, March 2017, acs.jctc.6b01181. https://doi.org/10.1021/acs.jctc.6b01181.
