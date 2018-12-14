import numpy as np
import mdtraj as md
import scipy

def _get_unit_vectors(vec):
    return vec/np.sqrt(np.einsum('ij,ij->i', vec, vec))[:,None]


def get_dye_centers(trj, dye_length=0.6):
    """Determines the idealized location of each residues beta-carbon
    coordinate from N, CA, and C coordinates.

    Attributes
    ----------
    trj : md.Trajectory,
        MDTraj object of a trj.

    Returns
    ----------
    cb_coords : array, shape=(n_residues,),
        A list of beta-carbon coordinates.
    """
    # get CA, N, and O coordinates for each residue
    top = trj.topology
    ca_iis = top.select("name CA")
    c_iis = top.select("name C")
    n_iis = top.select("name N")
    ca_coords = np.vstack(trj.xyz[:, ca_iis])
    c_coords = np.vstack(trj.xyz[:, c_iis])
    n_coords = np.vstack(trj.xyz[:, n_iis])
    # determine the vector normal to the plane defined by the points CA, N, 
    # and O. 
    norm_vec_1 = _get_unit_vectors(ca_coords - n_coords)
    norm_vec_2 = _get_unit_vectors(ca_coords - c_coords)
    norm_vec = _get_unit_vectors(np.cross(norm_vec_1, norm_vec_2))
    # determine the vector that points out from the CA (perpendicular to the
    # above vector.
    ca_vec = _get_unit_vectors(ca_coords - ((n_coords+c_coords)/2.))
    # get point along vector towards CB with length l
    theta = np.pi/6.
    ca_dist = np.sin(theta)*dye_length
    norm_dist = np.cos(theta)*dye_length
    center_coords = ca_coords + (ca_dist * ca_vec) + (norm_dist * norm_vec)
    # reshape to match n_frames
    center_coords = center_coords.reshape(trj.n_frames, ca_iis.shape[0], 3)
    return center_coords


def make_point_cloud(center=[0,0,0], r=1.0, ppa=10):
    single_line = ((np.arange(ppa) / (ppa-1)) - 0.5) * r * 2
    x, y ,z = np.meshgrid(single_line, single_line, single_line)
    square_cloud = np.array(list(zip(x.flatten(), y.flatten(), z.flatten())))
    dists = np.sqrt(np.einsum('ij,ij->i', square_cloud, square_cloud))
    iis = dists < r
    circle_cloud = square_cloud[iis] + center
    return circle_cloud


def _remove_touches_protein(
        xyz, dye_center, point_cloud, dye_bredth,
        touches_protein_cutoff=0.24):
    touches_protein_cutoff_sq = touches_protein_cutoff**2
    search_reduction = touches_protein_cutoff + dye_bredth
    diffs = xyz - dye_center
    atom_iis_to_check = np.where(
        np.einsum("ij,ij->i", diffs, diffs) < search_reduction**2)[0]
    atom_coords_to_check = xyz[atom_iis_to_check]
    for atom_ii in np.arange(atom_coords_to_check.shape[0]):
        diffs = point_cloud - atom_coords_to_check[atom_ii]
        dists_sq = np.einsum('ij,ij->i', diffs, diffs)
        point_cloud = point_cloud[dists_sq >= touches_protein_cutoff_sq]
    return point_cloud


def cluster_grids(point_cloud, spacing, n_clouds=all):
    orig_cluster_mapping= scipy.cluster.hierarchy.fclusterdata(
        point_cloud, t=spacing, criterion='distance')
    orig_cluster_mapping -= orig_cluster_mapping.min()
    largest_labels = np.argsort(-np.bincount(orig_cluster_mapping))
    if n_clouds is all:
        n_clouds = np.unique(orig_cluster_mapping).shape[0]
    contiguous_iis = np.hstack(
        [
            np.where(
                orig_cluster_mapping==label)[0]
            for label in largest_labels[:n_clouds]])
    contiguous_clouds = point_cloud[contiguous_iis]
    return contiguous_clouds


def reduce_point_cloud(
        xyz, dye_center, point_cloud, dye_bredth,
        ppa, touches_protein_cutoff=0.24, n_clouds=all):
    point_cloud = _remove_touches_protein(
        xyz=xyz, dye_center=dye_center, point_cloud=point_cloud,
        dye_bredth=dye_bredth, touches_protein_cutoff=touches_protein_cutoff)
    if point_cloud.shape[0] == 0:
        point_cloud = np.array([dye_center])
    else:
        point_cloud = cluster_grids(
            point_cloud, spacing=dye_bredth*2/ppa*1.5, n_clouds=n_clouds)
    return point_cloud


def _get_ensemble_FRET_distances(
        xyz, dye_centers, dye_bredth, ppa, n_clouds, base_point_cloud,
        min_dist):
    point_cloud_pair = [
        reduce_point_cloud(
            xyz,
            center_coord,
            base_point_cloud + center_coord,
            dye_bredth,
            ppa, n_clouds=n_clouds)
        for center_coord in dye_centers]
    return mean_distance_distribution(*point_cloud_pair, min_dist=min_dist)


def mean_distance_distribution(
        point_cloud1, point_cloud2, min_dist=0.6, max_warn=10000000):
    if point_cloud1.shape[0]*point_cloud2.shape[0] > max_warn:
        print("too many distances in point cloud!")
        raise
    if point_cloud1.shape[0] > point_cloud2.shape[0]:
        point_cloud_tmp = point_cloud1
        point_cloud1 = point_cloud2
        point_cloud2 = point_cloud_tmp
    all_dists = []
    for ii in np.arange(point_cloud1.shape[0]):
        diffs = point_cloud2 - point_cloud1[ii]
        dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        all_dists.append(dists)
    all_dists = np.hstack(all_dists)
    return all_dists[all_dists >= min_dist].mean()


def FRET_ensemble(
        trj, resSeq_pairs, dye_length=0.6, dye_bredth=2.0, ppa=12,
        n_clouds=2, min_dist=0.6):
    assert np.array(resSeq_pairs).shape[1] == 2
    dye_centers = get_dye_centers(trj, dye_length=dye_length)
    base_point_cloud = make_point_cloud(r=dye_bredth, ppa=ppa)
    prot_resSeqs = np.array([res.resSeq for res in trj.top.residues])
    all_FRET_mean_dists = []
    for resSeq_pair in resSeq_pairs:
        resis = np.array(
            [np.where(prot_resSeqs == resSeq)[0][0] for resSeq in resSeq_pair])
        FRET_ensemble_dists = []
        for frame_num in np.arange(trj.n_frames):
            FRET_ensemble_dists_tmp = _get_ensemble_FRET_distances(
                trj.xyz[frame_num], dye_centers[frame_num, resis], dye_bredth,
                ppa, n_clouds, base_point_cloud, min_dist)
            FRET_ensemble_dists.append(FRET_ensemble_dists_tmp)
        all_FRET_mean_dists.append(np.array(FRET_ensemble_dists))
    return all_FRET_mean_dists
