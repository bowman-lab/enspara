Pocket Detection
================

We've implemented the classic pocket detection algorithm ``LIGSITE`` in
``enspara``, and it can also be used to detect exposons.

Finding pockets with LIGSITE
----------------------------
    Enspara packages an implementaiton of the classic pocket-detection
    algorithm LIGSITE. LIGSITE builds a grid over the protein, and searches
    for concavities in the protein by extending a number of rays from each grid
    vertex, and then counts what fraction of them interact with protein. Points
    that are not inside the protein, but with rays that intersect the protein,
    are considered to be 'pocket vertices'.

.. code-block:: python
    
    import mdtraj as md
    from enspara import geometry

    pdb = md.load('reagents/m182t-a243-exposon-open.pdb')

    # run ligsite
    pockets_xyz = enspara.geometry.get_pocket_cells(struct=pdb)

    # build a pdb of hydrogen atoms for each grid point so it can be
    # examined in a visualization program (e.g. pymol)
    import pandas as pd

    top_df = pd.DataFrame()
    top_df['serial'] = list(range(pockets_xyz.shape[0]))
    top_df['name'] = 'PK'
    top_df['element'] = 'H'
    top_df['resSeq'] = list(range(pockets_xyz.shape[0]))
    top_df['resName'] = 'PCK'
    top_df['chainID'] = 0

    pocket_top = md.Topology.from_dataframe(top_df, np.array([]))
    pocket_trj = md.Trajectory(xyz=pockets_xyz, topology=pocket_top)
    pocket_trj.save('./pockets.pdb')
