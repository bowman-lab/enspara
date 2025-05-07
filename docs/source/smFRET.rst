smFRET predictions
======================

:code:`enspara` enables prediction of smFRET results from Markov State models.
Accordingly, you will need to have clustered your data (see  
:doc:`clustering </clustering>`,  
and built an appropriate MSM (see :doc:`Fitting-an-MSM </./tutorial/fitting>`).

You will also need to acquire either point clouds of MSMs of the dyes you wish
to model. We have deposited some dye MSMs `here <https://osf.io/82xtd/?view_only=b7f354e86eb144a69d9d047b42e21a9f>`_ for general use.

Once you are satisfied with your MSMs, you may predict smFRET results from them.

There are two levels of detail available for use:

1. :ref:`Apps <smFRET-app>`. smFRET prediction code is available in a 
command-line application that is capable of handling much of the bookkeeping
necessary.

2. Functions. The smFRET code is ultimately implemented as functions, which
offer the highest degree of control over the function's behavior, but also
require the most work on the user's part.


.. _smFRET-app:

smFRET App
--------------------------------

smFRET prediction functionality is availiable in :code:`enspara` in the script
:code:`apps/smFRET-dye-MC.py` or :code:`apps/smFRET-point-clouds.py`. 
Its help output explains at a granular level of detail what it is capable of, 
and so here we will seek to provide a high-level discussion of how it can be used.

When predicting smFRET, you will need to make a few important choices:

1. How would you like to treat the dyes? Treating dyes as point clouds is faster,
though does not account for dye lifetimes and requires choosing a Förster radius.
Accounting for dye dynamics (dye-MC) takes longer, but also provides dye-lifetimes
and is more physically realistic. 

2. Which dyes would you like to predict FRET for? You will need to acquire either
a dye MSM or a dye-point cloud for each dye of interest. We describe creation of 
additional dye MSMs 
`here <https://osf.io/82xtd/?view_only=b7f354e86eb144a69d9d047b42e21a9f>`_.



Predicting smFRET using dye MSMs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

smFRET calculations are parallelized using pool. It is generally fairly fast,
but MPI implementation has not been implemented yet. Scaling to many cluster centers
will make the calculation noticably slower. Predicting smFRET proceeds
in two steps, first modeling the dye lifetimes for each state in the protein MSM.

This can be done as follows:

.. code-block:: bash

    python /home/username/enspara/enspara/apps/smFRET_dye_MC.py calc_lifetimes\
    --donor_name 'AlexaFluor 488 B1R' \
    --acceptor_name 'AlexaFluor 647 C2R' \
    --donor_centers /path/to/donor/dye/MSM_centers.xtc \
    --donor_top /path/to/donor/dye/topology_file.pdb \
    --acceptor_centers /path/to/acceptor/dye/MSM_centers.xtc \
    --acceptor_top /path/to/acceptor/dye/topology_file.pdb \
    --donor_tcounts /path/to/donor/dye/transition/counts.npy \
    --acceptor_tcounts /path/to/acceptor/dye/transition/counts.npy \
    --dye_lagtime 0.002 \
    --prot_top /path/to/protein/topology_file.pdb \
    --prot_centers /path/to/protein/MSM/centers.xtc \
    --resid_pairs /path/to/residue/labeling/pairs.txt \
    --n_procs 40 \
    --n_samples 1000 \
    --output_dir /path/to/output/location/

This will calculate the dye-lifetimes for each residue pair in pairs.txt for each
protein center in the MSM, using the provided dye MSMs built with a lagtime of 2ps
(0.002 ns). In this case, 1000 Monte-Carlo simulations will be run for each protein center to generate a variety of dye lifetimes and emission events. 
We tend to see convergence of FRET efficiency for these
calculations with ~500-1000 events. :code:`pairs.txt` should look something like
this:
:code:`112 136` to calculate lifetimes for the dyes with labeling at positions
112 and 136, using the residue notation provided in :code:`topology_file.pdb` (resSeq).
To predict lifetimes for multiple dye pairs, simply append more resSeq positions
to :code:`pairs.txt`.

One file will be generated for each residue pair, with the format :code:`events-112-136.npy`

These events files will be a numpy array of shape (n_protein_centers, 2, n_events).
For each protein center, the time to dye emission will be saved along with the 
emission outcome. In this case, energy transfer = acceptor emission, 
radiative = donor emission, and non_radiative  = no observed emission.

Optionally, you can save out dye_trajectories and dye MSMs for each protein state 
with the commands
.. code-block:: bash
    --save_dtrj True \
    --save_dmsm True \

This will result in dye MSMs being saved in the output directory for each protein 
state. The dye MSMs will be similar to the original, though states that resulted 
in steric clashes will be removed and the remaining states rebuilt based on the 
original counts matrix.

The saved dtrj will be a ragged array of shape (n_bursts, variable). Each element 
in the ragged array corresponds to the cluster center number that the dye was in
at that step in the monte-carlo simulation. 


Next, you will need to simulate smFRET bursts.

This can be done as follows:

.. code-block:: bash

    python /home/username/enspara/enspara/apps/smFRET_dye_MC.py run_burst \
    --eq_probs /path/to/protein/eq_probs.npy \
    --t_counts /path/to/protein/t_counts.npy \
    --lifetimes_dir /path/to/dye/lifetimes/output \
    --lagtime 5 \
    --donor_name 'AlexaFluor 488 B1R' \
    --acceptor_name 'AlexaFluor 647 C2R' \
    --resid_pairs /path/to/residue/labeling/pairs.txt \
    --n_procs 2 \
    --output_dir /path/to/output/location/
    --correction_factor 10000 9000 8000 7000 6000 5000

This will run a kinetic monte carlo simulation to simulate smFRET bursts for
the provided Markov State Model and protein centers, using dye-labeling
on the residues specified in pairs.txt. The correction factor is a scaling factor
which slows the simulation timescale to match the experimental timescale. In this
case, we are calculating the FRET efficiency for a series of rescaling times.
Note- You must have calculated dye lifetimes for these residues or else 
the code will error. This calculation is generally very fast and is written 
mostly single threaded. Parallelization is supported across the number of 
dye pairs being calculated with pool. Lagtime is the protein MSM lagtime (in ns).
Optionally, you can provide your own interphoton times with the argument 
:code:`--photon_times /path/to/interphoton_times.npy`.


Three outputs will be created:

1. :code:`/path/to/output/FEs/FE-residue1-residue2-correction-factor.npy`. 
This is a numpy array of the simulated FRET efficiencies. Each entry is 
the FRET efficiency of a single photon burst. Typically, we histogram the
results and present these as the FRET efficency.

2. :code:`/path/to/output/Lifetimes/(a_or_d)_lifetimes-residue1-residue2-correction-factor.npy` is a 
numpy array of the acceptor or donor lifetimes. It will be a ragged array 
of shape (n_bursts, n_photons). 

3. :code:`/path/to/output/MSMs/residue1-donor_dye-residue2-acceptor_dye-(eqs or t_prbs).npy` is the modified protein MSM accounting for steric clashes that 
occured when the protein was labeled.

Predicting smFRET using dye point clouds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

smFRET calculations are parallelized using pool. It is generally fairly fast,
but MPI implementation has not been implemented yet. Scaling to many cluster centers
will make the calculation noticably slower. Predicting smFRET proceeds
in two steps, first modeling the inter-dye distance for each protein MSM state.

This can be done as follows:

.. code-block:: bash

    python /home/username/enspara/enspara/apps/smFRET-point-clouds.py model_dyes \
    /path/to/protein/MSM/centers.xtc \
    /path/to/protein/topology_file.pdb \
    /path/to/residue/labeling/pairs.txt
    --FRETdye1 /path/to/dye/pointcloud.pdb \ 
    --FRETdye2 /path/to/dye/pointcloud.pdb \
    --n_procs 40 \
    --output_dir /path/to/output/location/

This will calculate the pairwise distance distribution for each state
in the protein MSM, for each residue labeling pair in :code:`pairs.txt`. 

:code:`pairs.txt` should look something like
this:

.. code-block:: bash

    112 136
    145 223


This will calculate pairswise inter-dye distance distributions using the two
provided point clouds. We provide two point clouds for dyes AlexaFluor 488 and
AlexaFluor594 in enspara which are the default if no dyes are provided. 

Two files will be output:

1. :code:`/path/to/output/location/dye_distributions/bin_edges_residue1_residue2.h5`
which is a ragged array of shape (n_protein_centers, variable) and the bin edges of
the histogrammed distance distribution. The width of bins is fixed at 0.1Å.

2. :code:`/path/to/output/location/dye_distributions/probs_residue1_residue2.h5` is
the probability that dyes will be at the given distance associated with the above
histogram.


Next, you will need to simulate smFRET bursts.

This can be done as follows:

.. code-block:: bash

    python /home/username/enspara/enspara/apps/smFRET-dye-MC.py run_burst \
    /path/to/protein/MSM/eq_probs.npy \
    /path/to/protein/MSM/t_probs.npy \
    2 \
    /path/to/previous/output/dye_distributions \
    /path/to/residue/labeling/pairs.txt
    --n_procs 40 \
    --time_factor 1000 \
    --n_chunks 2 \
    --R0 5.4 \
    --output_dir /path/to/output/location

This will run a kinetic monte carlo simulation to simulate smFRET bursts for
the provided Markov State Model and protein centers, using dye-distance distributions
on the residues specified in pairs.txt. You must have calculated dye 
distance distributions for these residues or else the code will error.

In this case, we have built a MSM with a 2ns labtime, and believe the MSM is 1,000 
times "faster" than the experiment. We are calculating FRET efficiency using a Förster
radius of 5.4nm. One output file will be generated, 
:code:`FRET_E_residue1_residue2_time_factor.npy`, which is a numpy array of the FRET
efficiencies for each observed burst.
