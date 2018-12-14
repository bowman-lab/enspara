Clustering Trajectories
=======================

The first step to analyzing MD data is usually clustering. For simple to
moderately-complex clustering tasks, we make this pretty straightforward in
``enspara``.

With the :ref:`Clustering CLI <clustering-app>`, you can cluster the data like so:

.. code-block:: bash

    enspara cluster \
      --trajectories trajectory-*.xtc \
      --topology fs-peptide.pdb \
      --algorithm khybrid \
      --cluster-number 20 \
      --subsample 10 \
      --atoms '(name N or name C or name CA)' \
      --distances fs-khybrid-clusters0020-distances.h5 \
      --center-features fs-khybrid-clusters0020-centers.pickle \
      --assignments fs-khybrid-clusters0020-assignments.h5

This will cluster all the trajectories into 20 clusters using the k-hybrid
algorithm based on backbone (atoms named ``N``, ``CA`` or ``C``, per the
`MDTraj DSL <http://mdtraj.org/latest/atom_selection.html>`_) and output
the results (distance, center structures, and assignments) to files named
as specified on the command line (``fs-khybrid-clusters0020*``).

