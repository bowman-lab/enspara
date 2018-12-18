Tutorial
========

In this short tutorial, we'll walk you through a basic example using enspara
to make an MSM and do a simple analysis.

.. toctree::
    :maxdepth: 1

    clustering-trajectories
    fitting
    analysis

Before you get started, however, let's download some molecular dynamics data!
This MD data is of the Fs peptide (Ace-A_5(AAARA)_3A-NME), which is a fairly
common model system for studying protein folding. It was prepared by Robert
McGibbon for MSMBuilder3.

.. code-block:: bash

    mkdir fs_peptide && cd fs_peptide
    wget https://ndownloader.figshare.com/articles/1030363/versions/1 -O fs_peptide.zip
    unzip fs_peptide.zip

While you're waiting for that download, check the textbook on MSMs, "An
Introduction to Markov State Models and Their Application to Long Timescale
Molecular Simulation" edited by Greg Bowman, Vijay Pande, and Frank Noe. That book
describes the theoretical and empirical groundwork for a lot of what we'll do.

Anyway, once your download is finished, if you ``ls``, you should see a bunch
of trajectories (``*.xtc``), a pdb file ``fs-peptide.pdb``, and a few other
files. If you do, you're ready to move on.
