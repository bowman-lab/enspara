import itertools
import numpy as np
import mdtraj as md


def rmsf_calc(centers, populations=None, ref_frame=0, per_residue=True):
    """Calculated the population weighted RMSF from a frame in a MSM

    Attributes
    ----------
    centers : md.trajectory, shape=(n_states,),
        The cluster centers to use for RMSF calculations.
    populations : array-like, shape=(n_states,), default=None,
        The population of each state in the MSM. If not supplied,
        all frames are weighted equally.
    ref_frame : int, default=0,
        The reference state in the MSM to use for calculation
        deviations from. If not supplied, first frame is used.
    per_residue : bool, default=True,
        Optionally returns rmsf averaged over residues. If False, will
        return the rmsf per atom.

    Returns
    ----------
    rmsfs : nd.array, shape=(n_residues,),
        Returns the population weighted RMSF of each residue.
    """
    # align all states to reference frame
    centers = centers.superpose(centers[ref_frame])

    # if no populations are supplied, generate a uniform distribution
    if populations is None:
        populations = np.ones(centers.n_frames) / centers.n_frames

    # get differences between coordinates
    diffs = centers.xyz - centers.xyz[ref_frame]

    # dot product differences
    dists_per_atom_sq = np.einsum('ijk,ijk->ij', diffs, diffs)

    if per_residue:
        # obtain indices of all atoms partitioned by residues
        atom_iis_per_resi = np.array(
            [[a.index for a in r.atoms] for r in centers.top.residues])

        # average the dot products within each residue
        avg_resi_dists = np.array(
            [
                np.mean(dists_per_atom_sq[:, iis], axis=1)
                for iis in atom_iis_per_resi])

        # population weight the RMSFs
        rmsfs = np.sqrt((avg_resi_dists*populations).sum(axis=1))
    else:
        # population weight rmsfs per atom
        rmsfs = np.sqrt((dists_per_atom_sq*populations[:,None]).sum(axis=0))

    return rmsfs


def _bfactors_from_rmsfs(pdb, rmsfs):
    """Given a PDB and a list of RMSFs, returns a list of the rmsf
    values in the shape of all the atoms in the PDB"""
    bfactors = np.concatenate(
        [
            list(itertools.repeat(rmsf, r.n_atoms))
            for rmsf,r in zip(rmsfs, pdb.top.residues)])
    return bfactors
