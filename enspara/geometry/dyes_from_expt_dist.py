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
from scipy.stats import kurtosis, entropy, skew

def FRET_efficiency(dists, r0, offset=0):
    #Convert distance into FRET efficiency given a Forster radius (r0) and distance offset
    r06 = r0**6
    return r06 / (r06 + ((dists + offset)**6))


def make_distribution(probs, bin_edges):
    probs_norm = ra.RaggedArray([l/l.sum() for l in probs])
    dist_vals = (bin_edges[:,1:] + bin_edges[:,:-1]) / 2.
    dist_distribution = ra.RaggedArray(
        np.vstack([dist_vals._data, probs_norm._data]).T, lengths=probs_norm.lengths)
    return dist_distribution

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
            '%s is not a path to a pdb, have you tried using an ENSPARA provided dye?')
            #User should never see this error when using the app.
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


def sample_FE_probs(dist_distribution, states, R0):
    dists = []
    bin_width = dist_distribution[0][1,0] - dist_distribution[0][0,0]
    for state in states:
        #Introduce a new random seed in each location
        #otherwise pool with end up with the same seeds.
        np.random.seed()
        dist = np.random.choice(
            dist_distribution[state][:,0], p=dist_distribution[state][:,1])
        dist += (np.random.random()*bin_width) - (bin_width/2.)

        dists.append(dist)
    FEs = FRET_efficiency(np.array(dists), R0)
    return FEs


def _sample_FRET_histograms(
        MSM_frames, T, populations, dist_distribution, R0, n_photon_std):
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


    #Introduce a new random seed in each location otherwise pool with end up with the same seeds.
    rng=np.random.default_rng()

    # determine number of frames to sample MSM
    n_frames = np.amax(MSM_frames) + 1

    # sample transition matrix for trajectory
    initial_state = rng.choice(np.arange(T.shape[0]), p=populations)    

    trj = synthetic_trajectory(T, initial_state, n_frames)

    # get FRET probabilities for each excited state
    FRET_probs = sample_FE_probs(dist_distribution, trj[MSM_frames], R0)

    # flip coin for donor or acceptor emisions
    acceptor_emissions = rng.random(FRET_probs.shape[0]) <= FRET_probs

    # average for final observed FRET
    if n_photon_std is None:
        FRET_val = np.mean(acceptor_emissions)
        FRET_std = None
    else:
        # optionally chunk emissions and assess intraburst variation
        FRET_subsets = divide_chunks(acceptor_emissions, n_photon_std)
        FRET_chunks = [np.mean(subset) for subset in FRET_subsets]
        FRET_std = np.std(FRET_chunks)
        FRET_val = np.mean(acceptor_emissions)#np.mean(FRET_chunks)

    return FRET_val, FRET_std, trj


def sample_FRET_histograms(
    T, populations, dist_distribution, 
    MSM_frames, R0, n_procs=1, n_photon_std=None):
    """samples a MSM to regenerate experimental FRET distributions

    Attributes
    ----------
    T : array, shape=(n_states, n_states),
        Transition probability matrix.
    populations : array, shape=(n_states, )
        State populations.
    dist_distribution : ra.RaggedArray, shape=(n_states, None, 2)
        The probability of a fluorophore-fluorophore distance.
    MSM_frames : list of lists,
        A list of lists of times between photons in a given burst. 
        Each list should be it's own burst.
        Provide in microseconds.
    lagtime : float,
        MSM lagtime used to construct the transition probability
        matrix in nanoseconds.
    n_photon_std : int, default=None,
        The number of photons to chunk for assessing variation within a
        burst. Must be less than n_photons. Default: None will not
        assess the intraburst varaition.
    n_procs : int, default=1,
        Number of cores to use for parallel processing.
    R0: float, default=5.4,
        Forster radius for specified dyes

    Returns
    ----------
    FEs : nd.array, shape=(n_samples, 2),
        A list containing a FRET efficiency and an intraburst standard
        deviation for each drawing.
    E_Traj: ra.array, snape=(n_samples,2),
        A list containing a FRET efficiency and the center indicies used
        in the synthetic trajectory for each drawing.
    """

    if n_procs > 1:
        # fill in function values
        sample_func = partial(
            _sample_FRET_histograms, T=T, populations=populations,
            dist_distribution=dist_distribution, R0=R0, n_photon_std=n_photon_std)

        # multiprocess
        pool = Pool(processes=n_procs)
        FE= pool.map(sample_func, MSM_frames)
        pool.terminate()

    else:
        FE = [_sample_FRET_histograms(MSM_frame, T=T, populations=populations,
            dist_distribution=dist_distribution, R0=R0, n_photon_std=n_photon_std) 
            for MSM_frame in MSM_frames]

    # numpy the output
    FE= np.array(FE, dtype=object)
    FEs=FE[:,0:2]
    trajs=FE[:,2]

    return FEs, trajs

def convert_photon_times(inter_photon_times, lagtime, slowing_factor):
    #Take the inter_photon times (in us) and convert to MSM frame steps
    #Accounting for a slowing factor for the MSM.
    #Lagtime should be in ns.
    conversion_factor=1000/(lagtime*slowing_factor)

    #Multiply experimental wait times by this to get MSM steps.
    MSM_frames=np.array([np.cumsum(np.multiply(inter_photon_times[i], conversion_factor), dtype=int)
     for i in range(len(inter_photon_times))], dtype='O')
    return MSM_frames

def int_norm_hist(xs, ys):
    """simple integration normalization"""
    if ys.shape[0] == xs.shape[0] - 1:
        heights = ys
    elif ys.shape[0] == xs.shape[0]:
        heights = (ys[1:] + ys[:-1]) / 2.
    dx = xs[1:] - xs[:-1]
    I = np.sum(heights*dx)
    return ys/I

def histogram_to_match_expt(pred_data, expt_data):
    #Histograms and normalizes a predicted 1D np array
    #To match the experimental histogramming
    bin_centers=expt_data[:,0]
    bin_width=bin_centers[1]-bin_centers[0]
    lower_range=bin_centers[0]-(bin_width/2)
    upper_range=bin_centers[-1]+(bin_width/2)
    nbins=len(bin_centers)
    if np.ndim(pred_data)==1:
        counts, bin_edges = np.histogram(pred_data, range=[lower_range, upper_range], bins=nbins)
        probs=counts/counts.sum()
    else:
        probs=[]
        for i in range(len(pred_data)):
            temp_counts,bins=np.histogram(pred_data[i],range=[lower_range, upper_range], bins=nbins)
            probs.append(temp_counts/temp_counts.sum())
        probs=np.array(probs)
    return probs

def Sum_sq_resid(expt_data, pred_data):
    RSS=np.sum((pred_data-expt_data)**2, axis=1)
    return RSS

def normalize_array(array):
    if np.ndim(array)==1:
        norm_array=(array-np.amin(array))/(np.amax(array)-np.amin(array))
    else:
        norm_array=[]
        for i in range(len(array)):
            norm_array.append((array[i]-np.amin(array[i]))/(np.amax(array[i])-np.amin(array[i])))
    return norm_array

def remake_data_from_hist(histo_data):
    #Converts histogrammed data back to raw data. Will cause some shifting
    #Supply an array of shape [# bins, 2] where each subarray is [bin_center, bin_count]
    bin_centers=histo_data[:,0]
    bin_width=bin_centers[1]-bin_centers[0]
    lower_range=bin_centers[0]-(bin_width/2)
    upper_range=bin_centers[-1]+(bin_width/2)
    bin_counts=histo_data[:,1].astype(int)

    rebuilt_data=[]
    for i, bin_count in enumerate(bin_counts):
        rebuilt_data.append(np.random.uniform(
            low=bin_centers[i]-(bin_width/2),
            high=bin_centers[i]+bin_width/2, 
            size=int(bin_count)))

    rebuilt_data=np.array(list(np.concatenate(rebuilt_data).flat))
    return rebuilt_data

def calc_4_moments(histo_data):
    #Calculates the 4 moments of a histogram
    #Works on 1D or 2D arrays, for 2D calculates on axis=1
    if np.ndim(histo_data)==1:
        data_mean=np.mean(histo_data)
        data_std=np.std(histo_data)
        data_skew=skew(histo_data)
        data_kurtosis=kurtosis(histo_data, fisher=True)
        moments=np.vstack((data_mean,data_std,data_skew,data_kurtosis))
    else:
        data_mean=np.mean(histo_data, axis=1)
        data_std=np.std(histo_data, axis=1)
        data_skew=skew(histo_data, axis=1)
        data_kurtosis=kurtosis(histo_data, axis=1, fisher=True)
        moments=np.vstack((data_mean,data_std,data_skew,data_kurtosis))
    return moments
    
def calc_2_3_4_moments(histo_data):
    #Calculates the 4 moments of a histogram
    #Works on 1D or 2D arrays, for 2D calculates on axis=1
    if np.ndim(histo_data)==1:
        data_std=np.std(histo_data)
        data_skew=skew(histo_data)
        data_kurtosis=kurtosis(histo_data, fisher=True)
        moments=np.vstack((data_std,data_skew,data_kurtosis))
    else:
        data_std=np.std(histo_data, axis=1)
        data_skew=skew(histo_data, axis=1)
        data_kurtosis=kurtosis(histo_data, axis=1, fisher=True)
        moments=np.vstack((data_std,data_skew,data_kurtosis))
    return moments