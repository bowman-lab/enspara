import glob
import mdtraj as md
import numpy as np
import os
import scipy
from functools import partial
from multiprocessing import Pool
from ..msm.synthetic_data import synthetic_trajectory
from .. import ra
from ..exception import DataInvalid


def FRET_efficiency(dists, offset=0, r0=5.4):
    r06 = r0**6
    return r06 / (r06 + ((dists + offset)**6))


def load_dye(dye):
    """Loads a FRET dye point cloud.

    Attributes
    ----------
    dye : str,
        The path or name of a dye file. i.e. 'AF488'.

    Returns
    ----------
    dye_pdb : md.Trajectory,
        An MDTraj object representing the pdb of a FRET dye point cloud.
    """
    # obtain paths for dye folder and potential dye PDB file
    geometry_path = os.path.split(
        os.path.abspath(__file__))[0]
    dye_folder_path = os.path.join(
        os.path.split(geometry_path)[0], 'data', 'dyes')
    dye_path = os.path.join(dye_folder_path, '%s.pdb' % dye)
    # check if str supplied is a path to a file
    if os.path.exists(dye):
        dye_pdb = md.load(dye)
    # otherwise try and load from data folder
    elif os.path.exists(dye_path):
        dye_pdb = md.load(dye_path)
    # print error message
    else:
        dye_path_names = np.sort(glob.glob(os.path.join(dye_folder_path, '*.pdb')))
        dye_names = ", ".join(
            [
                os.path.split(p)[-1].split('.pdb')[0]
                for p in dye_path_names]) 
        raise DataInvalid(
            '%s is not a path to a pdb, nor does it exist in enspara. '
            'Consider using one of the following: %s' % (dye, dye_names))
    return dye_pdb

def norm_vec(vec):
    """Divides vector by its magnitude to obtain unit vector.
    """
    # depending on shape, gets list of unit vectors or single unit vector
    try:
        unit = vec / np.sqrt(np.einsum('ij,ij->i', vec, vec))[:,None]
    except:
        unit = vec / np.sqrt(np.dot(vec, vec))
    return unit


def divide_chunks(l, n):
    """Returns `n`-sized chunks from `l`.
    """
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


def int_norm(xs, ys):
    """Normalizes ys so that integral is unity.
    """
    dx = xs[1] - xs[0]
    I = np.sum(ys*dx)
    return (ys / I)


def determine_rot_mat(pdb, resSeq):
    """Determines the rotation matrix needed to align coordinates
    to a specific residue. Calculates rotation centered around CA,
    with CB pointing to the z-axis and N laying in the z-y plane.

    Attributes
    ----------
    pdb : md.Trajectory,
        MDTraj trajectory object containing the pdb coordinates and topology.
    resSeq : int,
        The residue sequence number to use for calculating rotation matrix.

    Returns
    ----------
    M : nd.array, shape=(3,3),
        The rotation matrix.
    ca_coord : nd.array, shape=(3, )
        The CA coordinate of the specified residue number.
    """
    # Determine the CB coordinate. Calculates where it should be if not
    # present (i.e. gly or pro).
    cb_coord = calc_cb_coords(pdb, resSeqs=resSeq)[0]
    # Extracts CA and N coordinates
    ca_coord = pdb.xyz[0, find_atom_index(pdb, resSeq, 'CA')]
    n_coord = pdb.xyz[0, find_atom_index(pdb, resSeq, 'N')]
    # z axis is in the direction of the CB coordinate from the CA coordinate
    z_vec = norm_vec(cb_coord - ca_coord)
    # x axis is normal to the z axis and the y axis (N coordinate lays on
    # this plane)
    x_vec = norm_vec(np.cross(norm_vec(n_coord - ca_coord), z_vec))
    # obtain y vector as orthonormal to other vectors
    y_vec = norm_vec(np.cross(z_vec, x_vec))
    # obtain rotation matrix
    M = np.array([x_vec, y_vec, z_vec])
    return M, ca_coord


def find_atom_index(pdb, resSeq, atom_name):
    """Helper function to determine the index of an atom
    with a specified resSeq and atom-name"""
    ii = None
    # iterate over residues
    for residue in pdb.top.residues:
        # if the correct resSeq, iterate over atoms
        if residue.resSeq == resSeq:
            for atom in residue.atoms:
                # if correct atom name, store index and break
                if atom.name == atom_name:
                    ii = atom.index
                    break
            else:
                continue
            break
    return ii


def calc_cb_coords(pdb, resSeqs=None):
    """Calculates the CB coordinates from CA, C, and N coordinates.

    Attributes
    ----------
    pdb : md.Trajectory,
        MDTraj trajectory object containing the pdb coordinates and topology.
    resSeqs : list, default=None,
        The residues to determine CB coordinates. If none are supplied, will
        calculate a CB coordinate for every residue.

    Returns
    cb_coords : ndarray, shape=(n_coordinates, 3),
        A CB coordinate for every residue supplied.
    """
    l = 0.153 # average CA-CB distance
    # get CA, N, and C coordinates for each residue
    top = pdb.topology
    # grab indices of coordinates
    if resSeqs is None:
        ca_iis = top.select("name CA")
        c_iis = top.select("name C")
        n_iis = top.select("name N")
    else:
        resSeqs = np.array(resSeqs).reshape(-1)
        ca_iis = np.array(
            [find_atom_index(pdb, r, 'CA') for r in resSeqs])
        c_iis = np.array(
            [find_atom_index(pdb, r, 'C') for r in resSeqs])
        n_iis = np.array(
            [find_atom_index(pdb, r, 'N') for r in resSeqs])
    ca_coords = pdb.xyz[0][ca_iis]
    c_coords = pdb.xyz[0][c_iis]
    n_coords = pdb.xyz[0][n_iis]
    # determine the vector normal to the plane defined by the points CA, N, 
    # and O.
    norm_vec_1 = norm_vec(ca_coords - n_coords)
    norm_vec_2 = norm_vec(ca_coords - c_coords)
    normed_vec = norm_vec(np.cross(norm_vec_1, norm_vec_2))
    # determine the vector that points out from the CA (perpendicular to the
    # above vector.
    ca_vec = norm_vec(ca_coords - ((n_coords+c_coords)/2.))
    # get point along vector towards CB with length l
    theta = np.pi/6.
    ca_dist = np.sin(theta)*l
    norm_dist = np.cos(theta)*l
    cb_coordinates = ca_coords + (ca_dist * ca_vec) + (norm_dist * normed_vec)
    return cb_coordinates


def rodrigues_rotation(v, k, theta, centers=None):
    """Applies Rodrigues' rotation on a coordinate trajectory.
    Vrot = v*cos(theta) + (k x v)sin(theta) + k(k.v)(1-cos(theta))

    Parameters
    ----------
    v : array, shape [n_frames, n_coordinates, dim_coordinate]
        The coordinates to rotate around a vector. Primarily used
        to rotate a trajectory of coordinates.
    k : array, shape [n_frames, dim_coordinate]
        A list of vectors to rotate each frame by individually.
    theta : float
        The angle to rotate by.
    centers : array, shape [n_frames, dim_coordinate]
        The center coordinate to rotate around.

    Returns
    ----------
    new_coords : array, shape [n_frames, n_coordinates, dim_coordinate]
        Updated coordinates: `v`, rotated around `k`, centered at
        `centers`, by `theta`.
    """
    if centers is None:
        centers = np.array([0,0,0])
    else:
        centers = centers[:, None, :]
    # center coordinates to prep for rotation
    v_centered = v - centers
    # calculate each of the three terms in the rodrigues rotation
    first_terms = v_centered * np.cos(theta)
    second_terms = np.cross(k[:, None, :], v_centered)*np.sin(theta)
    k_dot_vs = np.einsum('ijk,ijk->ij', k[:, None, :], v_centered)
    ang = 1- np.cos(theta)
    third_terms = np.array(
        [k[i]*k_dot_vs[i][:, None]*ang for i in range(len(k_dot_vs))])
    new_coords = (first_terms + second_terms + third_terms) + centers
    return new_coords


def _remove_touches_protein(coords, pdb, probe_radius=0.17):
    """Helper function for removing coordinates that are too close to a
    protein atom
    """
    # get distance cutoffs
    atomic_radii = np.array([a.element.radius for a in pdb.top.atoms])
    dist_cutoffs = (atomic_radii + probe_radius)
    # extract pdb coordinates
    pdb_xyz = pdb.xyz[0]
    # get all pairwise distances
    dists = scipy.spatial.distance.cdist(pdb_xyz, coords)
    # slice distances that are not near protein
    reduced_coords = coords[np.all(dists > dist_cutoffs[:,None], axis=0)]
    return reduced_coords


def remove_touches_protein(coords, pdb, probe_radius=0.17):
    """Remove coordinates that are too close to the protein.
    
    Attributes
    ----------
    coords : array, shape=(n_coordinates, 3),
        The coordinates to determine if touches protein.
    pdb : md.Trajectory,
        MDTraj trajectory object of the protein.
    probe_radius : float, default=0.17,
        Radius (in nm) for coordinate being too close to a protein atom's
        van der Waals radius. Default is 0.17, the radius of water.
    """
    # pairwise distances can add up! Chunks calculation if there would be
    # too many pairwise distances.
    # Set maximum number pairwise distances before deciding to chunk data
    max_dist_points = 5E7
    # if too many pairwise distances, chunk data
    if coords.shape[0]*pdb.xyz[0].shape[0] > max_dist_points:
        reduced_coords = np.zeros((0,3))
        # set chunking size
        chunk_size = 2048
        # chunk coordinate set
        coords_chunked = divide_chunks(coords, chunk_size)
        # obtain chunked histograms
        for coords_chunk in coords_chunked:
            # append chunked results
            reduced_coords = np.vstack(
                [
                    reduced_coords,
                    _remove_touches_protein(
                        coords_chunk, pdb, probe_radius=probe_radius)])
    else:
        # if no chunking requires, computes all at once
        reduced_coords = _remove_touches_protein(
            coords, pdb, probe_radius=probe_radius)
    return reduced_coords


def cluster_grids(point_cloud, spacing, n_clouds=all):
    """Clusters grid points and returns top volume clouds.

    Attributes
    ----------
    point_cloud : nd.array, shape=(n_coordinates, 3),
        The point cloud to cluster.
    spacing : float,
        Distance between points to consider within a cluster.
    n_clouds : int, default=all,
        The number of clusters to return.

    Returns
    ----------
    contiguous_clouds : nd.array, shape=(n_coordinates, 3),
        The coordinates of points within the top n_clouds
        after clustering.
    """
    # cluster using scipy hierarchical clustering
    orig_cluster_mapping = scipy.cluster.hierarchy.fclusterdata(
        point_cloud, t=spacing, criterion='distance')
    # sort based on number of points in cluster
    orig_cluster_mapping -= orig_cluster_mapping.min()
    largest_labels = np.argsort(-np.bincount(orig_cluster_mapping))
    # extract top n_clouds
    if n_clouds is all:
        n_clouds = np.unique(orig_cluster_mapping).shape[0]
    contiguous_iis = np.hstack(
        [
            np.where(
                orig_cluster_mapping==label)[0]
            for label in largest_labels[:n_clouds]])
    contiguous_clouds = point_cloud[contiguous_iis]
    return contiguous_clouds


def align_dye_to_res(pdb, dye_coords, resSeq):
    """Aligns dye point cloud to a residue.

    Attributes
    ----------
    pdb : md.Trajectory,
        The pdb to use for alignment.
    dye_coords : nd.array, shape=(n_coords, 3),
        The coordinates of the dye or dye point cloud to align.
    resSeq : int,
        The residue sequence number that dye will be aligned to.

    Returns
    ----------
    algined_coords : nd.array, shape=(n_coords, 3),
        The aligned coordinates.
    """
    M, t = determine_rot_mat(pdb, resSeq)
    aligned_coords = np.matmul(dye_coords, M) + t
    return aligned_coords


def bincount_dists(dists, bin_width=0.1):
    """Generates a histogram with a specific bin width
    """
    nbins = int(dists.max() / bin_width) + 2
    max_bin = nbins * bin_width
    counts, bin_edges = np.histogram(dists, bins=nbins, range=[0, max_bin])
    return counts, bin_edges


def pairwise_distance_distribution(coords1, coords2, bin_width=0.1):
    """Generate a probability distribution of all pairwise distances within
    two sets of coordinates. Returns distribution as a normalized histogram.

    Attributes
    ----------
    coords1 : nd.array, shape=(n_coords1, 3),
        The first set of coordinates to calculate pairwise distances.
    coords2 : nd.array, shape=(n_coords1, 3),
        The second set of coordinates to calculate pairwise distances.
    bin_width : float, default=0.1,
        The bin width for the resultant histogram.

    Returns
    ----------
    probs : array, shape=(n_bins, ),
        The probability of having a distance within each bin.
    bin_edges : array, shape=(n_bins + 1, ),
        The edges of each bin in the resultant histogram.
    """
    # pairwise distances can add up! Chunks calculation if there would be
    # too many pairwise distances.
    # determine maximum pairwise distances before deciding to chunk data
    max_dist_points = 5E7
    # determine if need to chunk
    if coords1.shape[0]*coords2.shape[0] > max_dist_points:
        # set chunking size
        chunk_size = 2048
        # determine which coords to chunk
        if coords1.shape[0] > coords2.shape[0]:
            max_coords = coords1
            min_coords = coords2
        else:
            max_coords = coords2
            min_coords = coords1
        # chunk larger coordinate set
        coords_chunked = divide_chunks(max_coords, chunk_size)
        # initialize histograms
        counts = []
        bin_edges = []
        # obtain chunked histograms
        for coords in coords_chunked:
            dists = scipy.spatial.distance.cdist(min_coords, coords)
            counts_tmp, bin_edges_tmp = bincount_dists(dists, bin_width)
            counts.append(counts_tmp)
            bin_edges.append(bin_edges_tmp)
        # combine histogram counts
        tot_counts, bin_edges = _merge_histograms(counts, bin_edges)
    else:
        # don't worry about chunking and just calculate distances and make
        # single histogram
        dists = scipy.spatial.distance.cdist(coords1, coords2)
        tot_counts, bin_edges = bincount_dists(dists, bin_width)
    # normalize counts
    probs = int_norm_hist(bin_edges, tot_counts)
    return probs, bin_edges


def _merge_histograms(counts, bin_edges, weights=None):
    """Merges histograms into a single histogram. Only supported
    for histograms with uniform bin-widths that start from zero.

    Attributes
    ----------
    counts : list, shape=(n_histograms, ),
        A list of histogram counts.
    bin_edges : list, shape=(n_histograms, ),
        A list of histogram bin-edges.
    weights : list, shape=(n_histograms, ), default=None,
        A list of weights to use for combining histograms.

    Returns
    ----------
    tot_counts : nd.array, shape=(n_bins, ),
        The counts of each bin in the resultant histogram.
    tot_bin_edges : nd.array, shape=(n_bins + 1, ),
        The edges of each bin in the resultant histogram.
    """
    # equal weight everything is weights are not supplied
    if weights is None:
        weights = np.ones(len(counts))
    else:
        weights = np.array(weights).reshape(-1)
    # determine number of bins in each histogram and pad to largest length
    lens = [c.shape[0] for c in counts]
    n_pads = np.max(lens) - lens
    padded_counts = np.array(
            [
                np.hstack([counts[n], np.zeros(n_pads[n], dtype=int)])
                for n in np.arange(n_pads.shape[0])])
    # weight counts and sum down rows to get total number of counts
    weighted_counts = padded_counts*weights[:, None]
    tot_counts = np.sum(weighted_counts, axis=0)
    # bin edges should be histogram with the maximum number of bins
    tot_bin_edges = bin_edges[np.argmax(lens)]
    return tot_counts, tot_bin_edges


def _dye_distance_distribution(
        pdb, dye1, dye2, resSeq_list, cluster_grid_points=False):
    """Obtains a probability distribution of all pairwise distances between
    FRET dye labeling positions.

    Attributes
    ----------
    pdb : md.Trajectory,
        PDB of protein conformation.
    dye1 : md.Trajectory,
        PDB of first FRET dye coordinates.
    dye2 : md.Trajectory,
        PDB of second FRET dye coordinates.
    resSeq_list : list, shape=(2, ),
        List of resSeq pair, (i.e. [45, 204])
    cluster_grid_points : bool, default=False,
        Optionally cluster dye point clouds and return largest cloud.

    Returns
    ----------
    probs : nd.array, shape=(n_bins, ),
        The probability of observing a specific distances between FRET dyes.
    bin_edges : nd.array, shape=(n_bins + 1, ),
        The edges of each bin in the resultant histogram.
    """
    resSeq1, resSeq2 = resSeq_list[0], resSeq_list[1]
    # rotate and translate dye point clouds to residues
    d1_r1 = align_dye_to_res(pdb, dye1.xyz[0], resSeq1)
    d1_r2 = align_dye_to_res(pdb, dye1.xyz[0], resSeq2)
    d2_r1 = align_dye_to_res(pdb, dye2.xyz[0], resSeq1)
    d2_r2 = align_dye_to_res(pdb, dye2.xyz[0], resSeq2)
    # remove the points that touch a protein
    d1_r1 = remove_touches_protein(d1_r1, pdb, probe_radius=0.2)
    d1_r2 = remove_touches_protein(d1_r2, pdb, probe_radius=0.2)
    d2_r1 = remove_touches_protein(d2_r1, pdb, probe_radius=0.2)
    d2_r2 = remove_touches_protein(d2_r2, pdb, probe_radius=0.2)
    # optionally cluster the grid points
    if cluster_grid_points:
        d1_r1 = cluster_grids(d1_r1, spacing=0.25, n_clouds=1)
        d1_r2 = cluster_grids(d1_r2, spacing=0.25, n_clouds=1)
        d2_r1 = cluster_grids(d2_r1, spacing=0.25, n_clouds=1)
        d2_r2 = cluster_grids(d2_r2, spacing=0.25, n_clouds=1)
    # histogram the pairwise distances
    probs1, bin_edges1 = pairwise_distance_distribution(d1_r1, d2_r2)
    probs2, bin_edges2 = pairwise_distance_distribution(d1_r2, d2_r1)
    # average the two histograms
    probs, bin_edges = _merge_histograms(
        [probs1, probs2], [bin_edges1, bin_edges2], weights=[0.5, 0.5])
    return probs, bin_edges


def dye_distance_distribution(
        trj, dye1, dye2, resSeq_list, cluster_grid_points=False,
        n_procs=1):
    """Obtains a probability distribution of all pairwise distances between
    FRET dye labeling positions over a trajectory.

    Attributes
    ----------
    trj : md.Trajectory,
        Trajectory of protein conformations.
    dye1 : md.Trajectory,
        PDB of first FRET dye coordinates.
    dye2 : md.Trajectory,
        PDB of second FRET dye coordinates.
    resSeq_list : list, shape=(2, ),
        List of resSeq pair, (i.e. [45, 204])
    cluster_grid_points : bool, default=False,
        Optionally cluster dye point clouds and return largest cloud.
    n_procs : int, default=1,
        The number of cores to use for calculation. Parallel over the number
        of frames in supplied trajectory.

    Returns
    ----------
    probs : nd.array, shape=(n_frames, ),
        The probability of observing a specific distances between FRET dyes.
    bin_edges : nd.array, shape=(n_frames, ),
        The edges of each bin in the resultant histogram.
    """
    func = partial(
        _dye_distance_distribution, dye1=dye1, dye2=dye2,
        resSeq_list=resSeq_list, cluster_grid_points=cluster_grid_points)
    pool = Pool(processes=n_procs)
    outputs = pool.map(func, trj)
    pool.terminate()
    probs = ra.RaggedArray([output[0] for output in outputs])
    bin_edges = ra.RaggedArray([output[1] for output in outputs])
    return probs, bin_edges


def sample_FE_probs(dist_distribution, states):
    dists = []
    bin_width = dist_distribution[0][1,0] - dist_distribution[0][0,0]
    for state in states:
        dist = np.random.choice(
            dist_distribution[state][:,0], p=dist_distribution[state][:,1])
        dist += (np.random.random()*bin_width) - (bin_width/2.)
        dists.append(dist)
    FEs = FRET_efficiency(np.array(dists))
    return FEs

def _sample_FRET_histograms(
        n_sample, T, populations, dist_distribution, photon_distribution,
        n_photons, lagtime, n_photon_std):
    """Helper function for sampling FRET distributions. Proceeds as 
    follows:
    1) generate a trajectory of n_frames, determined by the specified
       burst length.
    2) determine when photons are emitted by sampling the photon_distribution 
    3) use the FRET efficiencies per state to color the photons as either
       acceptor or donor fluorescence
    4) average acceptor fluorescence to obtain total FRET efficiency for
       the window
    """

    #Define the number of photon events observed
    photon_events_observed=np.random.exponential(size=1, scale=50).astype(int)
    while photon_events_observed < n_photons:
        photon_events_observed=np.random.exponential(size=1, scale=50).astype(int)

    # obtain frames that a photon is emitted
    photon_times = np.cumsum(
        photon_distribution(size=photon_events_observed))
    photon_frames = np.array(photon_times // lagtime, dtype=int)

    # determine number of frames to sample MSM
    n_frames = photon_frames.max() + 1

    # sample transition matrix for trajectory
    initial_state = np.random.choice(np.arange(T.shape[0]), p=populations)
    trj = synthetic_trajectory(T, initial_state, n_frames)

    # get FRET probabilities for each excited state
    FRET_probs = sample_FE_probs(dist_distribution, trj[photon_frames])

    # flip coin for donor or acceptor emisions
    acceptor_emissions = np.random.random(FRET_probs.shape[0]) <= FRET_probs

    # average for final observed FRET
    if n_photon_std is None:
        FRET_val = np.mean(acceptor_emissions)
        FRET_std = None
    else:
        # optionally chunk emissions and assess intraburst variation
        FRET_subsets = divide_chunks(acceptor_emissions, n_photon_std)
        FRET_chunks = [np.mean(subset) for subset in FRET_subsets]
        FRET_std = np.std(FRET_chunks)
        FRET_val = np.mean(FRET_chunks)

    return FRET_val, FRET_std


def sample_FRET_histograms(
        T, populations, dist_distribution, photon_distribution,
        n_photons, lagtime, n_photon_std=None, n_samples=1, n_procs=1):
    """samples a MSM to regenerate experimental FRET distributions

    Attritbues
    ----------
    T : array, shape=(n_states, n_states),
        Transition probability matrix.
    populations : array, shape=(n_states, )
        State populations.
    dist_distribution : ra.RaggedArray, shape=(n_states, None, 2)
        The probability of a fluorophore-fluorophoe distance.
    photon_distribution : func,
        A callable function that samples from a distribution of
        photon wait-times, i.e. 'np.random.exponential'
    n_photons : int,
        The number of photons in a burst.
    lagtime : float,
        MSM lagtime used to construct the transition probability
        matrix in nanoseconds.
    n_photon_std : int, default=None,
        The number of photons to chunk for assessing variation within a
        burst. Must be less than n_photons. Default: None will not
        assess the intraburst varaition.
    n_samples : int, default=1,
        The number of times to sample FRET distribution.
    n_procs : int, default=1,
        Number of cores to use for parallel processing.

    Returns
    ----------
    FEs : nd.array, shape=(n_samples, 2),
        A list containing a FRET efficiency and an intraburst standard
        deviation for each drawing.
    """

    # fill in function values
    sample_func = partial(
        _sample_FRET_histograms, T=T, populations=populations,
        dist_distribution=dist_distribution,
        photon_distribution=photon_distribution,
        n_photons=n_photons, lagtime=lagtime,
        n_photon_std=n_photon_std)

    # multiprocess
    pool = Pool(processes=n_procs)
    FEs = pool.map(sample_func, np.arange(n_samples))
    pool.terminate()

    # numpy the output
    FEs = np.array(FEs)

    return FEs


def kinetic_avg_distributions(
        probs, bin_edges, tprobs, n_steps=1, n_samples=1):
    """Performs a kinetic averaging across distributions. For each state
    distribution, this function will return the observed distribution when
    drawing n_samples over n_steps, where the distribution drawn from evolves
    according to tprobs.

    Attributes
    ----------
    probs : array, shape=(n_states, n_bins),
        The probability of observing a particular value (i.e. the y-axis in a
        histogram), provided for each state.
    bin_edges : array, shape=(n_bins + 1, ),
        The bounds for the bins for each of the distributions. The probs
        should all share these bin_edges.
    tprobs : array, shape=(n_states, n_states),
        The transition probability matrix used for performing the kinetic
        averaging.
    n_steps : int, default=1,
        The final number of steps to take for use when averaging
        distributions.
    n_samples : int, default=1,
        The number of samples to draw along averaging.
    
    Returns
    --------
    kavg_probs : array, shape=(n_states, n_bins),
        The kinetic averaged probability of observing a particular value
        given a trajectory of n_steps, with n_samples observations, having
        started in a particular state.
        
    """
    # determine bin centers, which are used for averaging bin values
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.
    
    # determine sample numbers within n_steps
    sample_nums = np.array(
        np.linspace(
            start=0, stop=n_steps, num=n_samples, endpoint=True) + 1, dtype=int)

    # copy the probability distributions
    new_probs = np.copy(probs)
    
    # iterate over number of samples to average
    for n0 in np.arange(sample_nums.shape[0]):
        
        # obtain a time point
        sample_t = sample_nums[n0]
        
        # propagate transition probability matrix
        tprobs_propped = np.linalg.matrix_power(tprobs, sample_t)
        
        # obtain averaged distributions
        avg_probs = np.matmul(tprobs_propped, probs)
        
        # update probabilities of observing a particular distance
        probs_mats = np.einsum('ij,ik->ijk', new_probs, avg_probs)
        
        # determine what the previously mentioned distances are 
        dists_mat = np.sum(np.meshgrid(bin_centers, bin_centers*n0), axis=0) / (n0+1)

        # zero out new probs and refill below
        new_probs = np.zeros(probs.shape)

        # obtain new distributioins by iterating over the labels
        for l in np.arange(bin_centers.shape[0]):
            
            # obtain distance to label center
            abs_dists = np.abs(dists_mat - bin_centers[l])
            
            # determine distances that are within a bin's reach
            bin_widths = bin_edges[1:] - bin_edges[:-1]
            mask = abs_dists < bin_widths[l]
            
            # weight population linearly by distance to label center
            ps = abs_dists[mask]*(-1/bin_widths[l]) + 1
            
            # update label with weighted values
            new_probs[:, l] = np.sum(probs_mats[:,mask] * ps, axis=1)
            
    kavg_probs = new_probs
    return kavg_probs


def shot_noise(Em, NT):
    sig_sn = np.zeros(Em.shape[0])
    inner_prod = (Em*(1-Em))/NT
    iis = np.where(inner_prod > 0)
    sig_sn[iis] = np.sqrt(inner_prod[iis])
    return sig_sn


def int_norm_hist(xs, ys):
    """simple integration normalization"""
    if ys.shape[0] == xs.shape[0] - 1:
        heights = ys
    elif ys.shape[0] == xs.shape[0]:
        heights = (ys[1:] + ys[:-1]) / 2.
    dx = xs[1:] - xs[:-1]
    I = np.sum(heights*dx)
    return ys/I

def distribution_with_shot_noise(xs, data, heights, NT=50):
    sig_sn = shot_noise(data, NT)
    ys = np.zeros(xs.shape)
    for n in np.arange(sig_sn.shape[0]):
        if sig_sn[n] > 0:
            c = sig_sn[n]
            b = data[n]
            a = heights[n]/np.sqrt(2*np.pi*(c**2))
            new_gaussian = a*np.exp(-((xs-b)**2)/(2*(c**2)))
            ys += new_gaussian
    ys = int_norm(xs, ys)
    return ys


def pad_distributions(probs, bin_edges):
    if not isinstance(probs, ra.RaggedArray):
        probs = ra.RaggedArray(probs)
    if not isinstance(bin_edges, ra.RaggedArray):
        bin_edges = ra.RaggedArray(bin_edges)
    delta_lengths = probs.lengths.max() - probs.lengths
    padded_probs = np.array(
        [
            np.hstack(
                [probs[n0],
                 np.zeros(delta_lengths[n0])])
            for n0 in np.arange(probs.shape[0])])
    bin_edges = bin_edges[bin_edges.lengths.argmax()]
    return padded_probs, bin_edges


def change_bins(bin_edges, probs, bin_edges_new):
    dists_new = (bin_edges_new[1:] + bin_edges_new[:-1])/2.
    bin_labels_left = np.digitize(dists_new, bin_edges, right=False)
    bin_labels_right = np.digitize(dists_new, bin_edges, right=True)
    base_probs_new = probs[:, bin_labels_right - 1] + probs[:, bin_labels_left - 1]
    base_probs_new /= base_probs_new.sum(axis=1)[:,None]
    return base_probs_new, bin_edges_new


def FRET_distribution_from_distances(
        probs, bin_edges, tprobs=None, populations=None, n_steps=1,
        n_samples=1, r0=5.4, n_FRET_bins=100, n_plot_points=500,
        x_plot_range=[-0.2, 1.2]):

    if tprobs is None:
        tprobs = np.eye(len(probs))

    if populations is None:
        populations = np.ones(tprobs.shape[0]) / tprobs.shape[0]

    # zero pad distributions
    padded_probs, padded_edges = pad_distributions(probs, bin_edges)

    # convert distance bins to FRET efficiency
    FEs_bin_edges = FRET_efficiency(padded_edges, r0=r0)

    # adjust bin widths to be uniform between 0 and 1
    new_bins = np.linspace(start=0, stop=1, num=n_FRET_bins, endpoint=True)
    FEs_probs, FEs_bin_edges_new = change_bins(
        FEs_bin_edges[::-1], padded_probs[:,::-1], new_bins)

    # normalize probabilities to sum to 1 for each distribution
    FEs_probs /= FEs_probs.sum(axis=1)[:,None]

    # perform kinetic averaging
    FE_probs_kavg = kinetic_avg_distributions(
        FEs_probs, FEs_bin_edges_new, tprobs, n_steps=n_steps,
        n_samples=n_samples)

    # population weight kinetic averages
    FE_probs_avg = np.sum(FE_probs_kavg * populations[:,None], axis=0)

    # add shot noise
    xs_sn = np.linspace(
        start=x_plot_range[0], stop=x_plot_range[1], num=n_plot_points,
        endpoint=True)
    FEs_centers = (FEs_bin_edges_new[1:] + FEs_bin_edges_new[:-1]) / 2.
    ys_sn = distribution_with_shot_noise(
        xs_sn, FEs_centers, FE_probs_avg, NT=n_samples)
    return xs_sn, ys_sn
