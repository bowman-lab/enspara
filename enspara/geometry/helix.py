import numpy as np
from enspara import exception


def calculate_piecewise_helix_vectors(
        trj, helix_resnums=None, helix_start=None, helix_end=None):
    """Calculates the vectors along specified alpha-helices for each
    frame in a trajectory. Vectors are in the direction of the starting
    residue to the ending residue.
    Parameters
    ----------
    trj : md.Trajectory object
        An MDTraj trajectory object containing frames of structures to
        compute helix-vectors from.
    helix_resnums : array, shape [n_residues, ], optional, default: None
        A list of residues that correspond to an alpha-helix. This is
        useful if residue numbers within a helix are unordinary. If a
        list of residues is not supplied, a start and stop residue can
        be specified.
    helix_start : int, optional, default: None
        The starting residue of the helix.
    helix_start : int, optional, default: None
        The ending residue of the helix.
    Returns
    ----------
    vectors : array, [n_frames, 3]
        A list of unit-vectors corresponding to the direction of the
        specified alpha-helix for each frame in the trajectory.
    center_coords : array, [n_frames, 3]
        Each center coordinate of the helix-atoms. Can be used to
        reconstruct a line going through the alpha-helix.
    """
    if (helix_resnums is None) and ((helix_start is None) or
                                    (helix_end is None)):
        raise exception.ImproperlyConfigured(
            "Either 'helix_resnums' or 'helix_start' and 'helix_end' "
            "are required.")
    elif helix_resnums is None:
        helix_resnums = np.arange(helix_start, helix_end+1)
    top = trj.topology
    backbone_nums = _get_backbone_nums(top, helix_resnums)
    backbone_coords = trj.xyz[:, backbone_nums]
    vectors = _generate_vectors_from_coords(backbone_coords, n_avg=12)
    center_coords = backbone_coords.mean(axis=1)
    return vectors, center_coords


def calculate_summary_helix_vectors(
        trj, res_refs, helix_resnums=None, helix_start=None, helix_end=None):
    """Gets vector orientations and center points of an alpha helix
    relative to an alpha-carbon on a residue within the helix.

    Parameters
    ----------
    trj : md.Trajectory object
        An MDTraj trajectory object containing frames of structures to
        compute helix-vectors from.
    res_refs : array-like
        Residue ids (resSeq) for which to build a coordinate frame relative
        to the helical axis.
    helix_resnums : array, shape [n_residues, ], optional, default: None
        A list of residues that correspond to an alpha-helix. This is
        useful if residue numbers within a helix are unordinary. If a
        list of residues is not supplied, a start and stop residue can
        be specified.
    helix_start : int, optional, default: None
        The starting residue of the helix.
    helix_start : int, optional, default: None
        The ending residue of the helix.

    Returns
    ----------
    helix_vectors : array, shape [n_frames, 3]
        A list of unit-vectors corresponding to the direction of the
        specified alpha-helix for each frame in the trajectory.
    ref_vectors : array, shape [n_refs, n_frames, 3]
        A list of vectors that are orthogonal to the helix vector that
        passes through the alpha-carbon of each reference residue.
    cross_vectors : array, shape [n_refs, n_frames, 3]
        A list of vectors that are orthogonal to the helix vector and
        the ref_vectors.
    center_coords : array, [n_frames, 3]
        Each center coordinate of the helix-atoms. Can be used to
        reconstruct a line going through the alpha-helix.
    """
    top = trj.topology
    atom_refs = _get_CA_nums(top, res_refs)
    helix_vectors, helix_centers = calculate_piecewise_helix_vectors(
        trj, helix_resnums=helix_resnums, helix_start=helix_start,
        helix_end=helix_end)
    ref_points = trj.xyz[:, atom_refs]
    ref_vectors = _get_ref_vectors(helix_vectors, helix_centers, ref_points)
    cross_vectors = np.cross(ref_vectors, helix_vectors)
    return helix_vectors, ref_vectors, cross_vectors, helix_centers


def angles_from_plane_projection(vectors, v1, v2, degree=True):
    projection1 = np.einsum('ij,ij->i', vectors, [v1])
    projection2 = np.einsum('ij,ij->i', vectors, [v2])
    projection_vector = np.array(list(zip(projection1, projection2)))
    mags = np.sqrt(np.einsum('ij,ij->i', projection_vector, projection_vector))
    dot_prods = np.einsum('ij,ij->i', projection_vector, [[1,0]])
    inner_prod = dot_prods/mags
    angles = np.arccos(np.around(inner_prod, 5))
    iis_neg = np.where(projection2 < 0)
    angles[iis_neg] = -angles[iis_neg]
    if degree:
        angles *= 360./(2*np.pi)
    return angles, mags


def angles_from_vecs(vecs, to=0):
    """Compute the angle from one vector to all other vectors.

    Parameters
    ----------
    vecs : np.ndarray, shape=(n_vectors, 3)
        Vectors to compute the dot product of.
    to : int
        Index to compute the dot product of all `vecs` to.

    Returns
    -------
    angles : np.ndarray, shape=(n_vectors,)
        Angles between each vector in `vecs` and `to`.
    """

    mags = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
    dot_prods = np.einsum('ij,ij->i', vecs, [vecs[to]])
    inner_prod = dot_prods / mags[to] / mags
    angles = np.arccos(np.around(inner_prod, 5))
    return angles


def _get_unit_vectors(vecs):
    """Normalizes a row of vectors to unit magnitude"""
    mags = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
    return vecs/mags[:,None]


def __generate_stacked_averages(coords, n_avg=4):
    """Helper function for computing vectors from helix coordinates"""
    # average coords
    stacked_coords = np.hstack(coords)
    avg_coords_stacked = np.array(
        [
            np.mean(stacked_coords[num:num+n_avg], axis=0)
            for num in range(len(coords[0])-n_avg-1)])
    return avg_coords_stacked


def _generate_vectors_from_coords(coords, n_avg=4):
    """Computes vectors along an alpha-helix given backbone
    coordinates. Generates a running average of coordinate position
    and averages vectors between sequential average points. Returns
    a vector in the direction of the alpha-helix.
    Parameters
    ----------
    coords : array, shape [n_frames, n_coords, 3]
        An array containing n_frames, where each frame is a list of
        [x,y,z] coordinates corresponding to a helixs' backbone.
    n_avg : int, optional, default: 4
        The number of coordinates to compute a running average.
        If coordinates correspond to C-alphas, this value should be 4.
        If coordinates correspond to N-CA-C atoms, this value should be
        12 to encompass a full repeat.
    Returns
    ----------
    unit_vectors : array, shape [n_frames, 3]
        The unit-vector corresponding to the direction of the helix for
        each frame in coords.
    """
    avg_coords_stacked = __generate_stacked_averages(coords, n_avg)
    # average coords
    avg_vectors_stacked = np.mean(
        np.array(
            [
                np.subtract(avg_coords_stacked[num], avg_coords_stacked[num+1])
                for num in range(len(avg_coords_stacked)-1)]),
        axis=0)
    avg_vectors = np.array(
        [
            avg_vectors_stacked[num:num+3]
            for num in range(len(avg_vectors_stacked))[::3]])
    unit_vectors = _get_unit_vectors(avg_vectors)
    return unit_vectors


def _get_backbone_nums(top, resnums):
    """Returns the atom indices that correspond to N, CA, and C for a
    list a residues."""
    backbone_nums = np.concatenate(
        [
            [
                top.select("resSeq " + str(res)+" and name N")[0],
                top.select("resSeq " + str(res)+" and name CA")[0],
                top.select("resSeq " + str(res)+" and name C")[0]]
            for res in np.sort(resnums)])
    return backbone_nums


def _get_CA_nums(top, resnums):
    CAs = np.array(
        [
            top.select("resSeq " + str(res) + " and name CA")[0]
            for res in resnums])
    return CAs


def _get_ref_vectors(normal_vecs, vec_points, ref_points):
    a_m_p = vec_points[:, None, :] - ref_points
    a_m_p_dot_n = np.einsum('ijk,ijk->ij', a_m_p, normal_vecs[:,None,:])
    ref_vectors = np.array(
        [
            _get_unit_vectors(
                a_m_p[:,i,:] - normal_vecs*a_m_p_dot_n[:,i][:,None])
            for i in range(a_m_p.shape[1])])
    return ref_vectors
