# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import mdtraj as md
import numpy as np
import scipy.cluster.hierarchy

from joblib import Parallel, delayed


def _grid_to_xyz(grid):
    """Convert a grid object (grid[x_ind,y_ind,z_ind]=[x,y,z])
    to an array of x,y,z coordinates
    """

    n_cells = grid.shape[0] * grid.shape[1] * grid.shape[2]
    xyz = grid.reshape((n_cells, 3))

    return xyz


def xyz_to_mdtraj(xyz, cluster_ids=None):
    """Convert a set of x,y,z coordinates to and mdtraj.Trajectory with a carbon
    atom centered at each of the specified coordinates.

    Each carbon will be part of a residue called POK. If cluster_ids are
    specified, these will be used as the residue numbers ofr sets of carbons
    in the same cluster.

    Parameters
    ----------
    xyz : np.ndarray, shape=(n_atoms, 3)
        Cartesian coordinates to center carbons at.
    cluster_ids : np.ndarray, shape=(n_atoms)
        If specified, the numbers in this array (one corresponding to each
        carbon atom to be created) will become the residue numbers for each
        carbon.

    Returns
    -------
    struct : mdtraj.Trajectory
        A Trajectory with a single frame containing a carbon at each of the
        specified x,y,z coordinates with residue numbers determined bye the
        cluster_ids, if specified.
    """

    n_xyz = xyz.shape[0]
    element = md.element.carbon
    top = md.Topology()
    chain = top.add_chain()
    if cluster_ids is None:
        res = top.add_residue("POK", chain, 0)
        for i in range(n_xyz):
            top.add_atom('C', element, res)
        sorted_xyz = xyz
    else:
        sorted_xyz = np.zeros((n_xyz,3))
        inds_sorted_by_cluster = np.argsort(cluster_ids)
        prev_res_ind = -1
        for i in range(n_xyz):
            cur_res_ind = cluster_ids[inds_sorted_by_cluster[i]]
            if cur_res_ind != prev_res_ind:
                res = top.add_residue("POK", chain, cur_res_ind)
                prev_res_ind = cur_res_ind
            top.add_atom('C', element, res)
            sorted_xyz[i] = xyz[inds_sorted_by_cluster[i]]

    struct = md.Trajectory(sorted_xyz, top)

    return struct


def create_grid(struct, grid_spacing, padding=0):
    """Create a grid spanning a structure with cubic cells, where each
    edge is grid_spacing long.

    Parameters
    ----------
    struct : mdtraj.Trajectory
        Only the first frame of this Trajectory will be used.
    grid_spacing : float (nm)
        The length of each edge of a cell in nm. So a cell has a volume
        of grid_spacing^3.
    padding : int, default=0
        The number of grid points to pad on each side of the protein
        (for good measure).

    Returns
    -------
    grid : np.ndarray, shape=(n_x,n_y,n_z,3)
        An n_x by n_y by n_z grid with the Cartesian coordinates of each cell
        in the grid (grid[x_ind,y_ind,z_ind]=[x,y,z]).
    """

    x_min = struct.xyz[0,:,0].min()
    x_max = struct.xyz[0,:,0].max()
    y_min = struct.xyz[0,:,1].min()
    y_max = struct.xyz[0,:,1].max()
    z_min = struct.xyz[0,:,2].min()
    z_max = struct.xyz[0,:,2].max()

    n_x_cells = int(np.ceil((x_max-x_min)/grid_spacing)) + padding*2
    n_y_cells = int(np.ceil((y_max-y_min)/grid_spacing)) + padding*2
    n_z_cells = int(np.ceil((z_max-z_min)/grid_spacing)) + padding*2

    x_coords = (x_min - grid_spacing*padding) + np.arange(n_x_cells)*grid_spacing
    y_coords = (y_min - grid_spacing*padding) + np.arange(n_y_cells)*grid_spacing
    z_coords = (z_min - grid_spacing*padding) + np.arange(n_z_cells)*grid_spacing
    y_mesh, x_mesh, z_mesh = np.meshgrid(y_coords, x_coords, z_coords)
    grid = np.concatenate(
        [
            x_mesh[:, :, :, None],
            y_mesh[:, :, :, None],
            z_mesh[:, :, :, None]], axis=3)
    return grid


def _get_cell_inds_within_cutoff(grid, point, distance_cutoff):
    """Find the indices of all the cells of a grid that are within
    distance_cutoff of the specified point.
    """

    n_x_cells, n_y_cells, n_z_cells = grid.shape[:3]
    x_min, y_min, z_min = grid[0,0,0]
    x_max, y_max, z_max = grid[-1,-1,-1]
    grid_spacing = (x_max-x_min)/(n_x_cells-1)

    # get indices of cell that specified point falls in
    x_cell_ind = int((point[0]-x_min)/grid_spacing)
    y_cell_ind = int((point[1]-y_min)/grid_spacing)
    z_cell_ind = int((point[2]-z_min)/grid_spacing)

    # get block of cells within cutoff distance of point
    num_cells_cutoff = int(np.ceil(distance_cutoff/grid_spacing))
    # include an extra cell (e.g. extra +/-1) just to be safe
    min_x_cell_ind = np.max([0, x_cell_ind-num_cells_cutoff])
    max_x_cell_ind = np.min([n_x_cells-1, x_cell_ind+num_cells_cutoff])
    min_y_cell_ind = np.max([0, y_cell_ind-num_cells_cutoff])
    max_y_cell_ind = np.min([n_y_cells-1, y_cell_ind+num_cells_cutoff])
    min_z_cell_ind = np.max([0, z_cell_ind-num_cells_cutoff])
    max_z_cell_ind = np.min([n_z_cells-1, z_cell_ind+num_cells_cutoff])

    return min_x_cell_ind, max_x_cell_ind, min_y_cell_ind, max_y_cell_ind, min_z_cell_ind, max_z_cell_ind


def _check_cartesian_axis(touches_protein, rank):
    """Finds cells along the x-axis of a grid that are not filled by protein
    atoms (touches_protein[x_ind,y_ind,z_ind]=0) but are surrounded by protein
    atoms (touches_protein[x_ind,y_ind,z_ind]=1) and increments their rank
    (rank[x_ind,y_ind,z_ind]) by one.
    """

    n_x_cells, n_y_cells, n_z_cells = touches_protein.shape
    for j in range(n_y_cells):
        for k in range(n_z_cells):
            x = touches_protein[:,j,k]
            inds_touching_protein = np.where(x>0)[0]
            # make sure there are at least two cells containing protein with some space between
            if inds_touching_protein.shape[0] > 1 and inds_touching_protein[0]+1 < inds_touching_protein[-1]:
                inds_consider = np.arange(inds_touching_protein[0]+1, inds_touching_protein[-1])
                inds_surrounded_by_protein = inds_consider[np.where(x[inds_consider]==0)[0]]
                if inds_surrounded_by_protein.shape[0] > 0:
                    rank[inds_surrounded_by_protein,j,k] += 1


def _check_diagonal_axis_helper(touches_protein, rank):
    """Finds cells along a diagonal of a grid that are not filled by protein
    atoms (touches_protein[x_ind,y_ind,z_ind]=0) but are surrounded by protein
    atoms (touches_protein[x_ind,y_ind,z_ind]=1) and increments their rank
    (rank[x_ind,y_ind,z_ind]) by one.
    """

    n_x_cells, n_y_cells, n_z_cells = touches_protein.shape
    for i in range(n_x_cells-1):
        for j in range(n_y_cells-1):
            n_b4_edge = np.min((n_x_cells-i,n_y_cells-j,n_z_cells))
            x_inds = np.arange(i,i+n_b4_edge)
            y_inds = np.arange(j,j+n_b4_edge)
            z_inds = np.arange(n_b4_edge)
            diag = touches_protein[x_inds,y_inds,z_inds]
            inds_touching_protein = np.where(diag>0)[0]
            # make sure there are at least two cells containing protein with some space between
            if inds_touching_protein.shape[0] > 1 and inds_touching_protein[0]+1 < inds_touching_protein[-1]:
                inds_consider = np.arange(inds_touching_protein[0]+1, inds_touching_protein[-1])
                inds_surrounded_by_protein = inds_consider[np.where(diag[inds_consider]==0)[0]]
                if inds_surrounded_by_protein.shape[0] > 0:
                    x_ind = x_inds[inds_surrounded_by_protein]
                    y_ind = y_inds[inds_surrounded_by_protein]
                    z_ind = z_inds[inds_surrounded_by_protein]
                    rank[x_ind, y_ind, z_ind] += 1


def _check_diagonal_axis(touches_protein, rank):
    """Finds cells along a diagonal of a grid that are not filled by protein
    atoms (touches_protein[x_ind,y_ind,z_ind]=0) but are surrounded by protein
    atoms (touches_protein[x_ind,y_ind,z_ind]=1) and increments their rank
    (rank[x_ind,y_ind,z_ind]) by one.
    """

    _check_diagonal_axis_helper(touches_protein, rank)
    # swap axes to get other two faces
    # use sub-indices to avoid double/triple counting diagonals
    _check_diagonal_axis_helper(
        touches_protein.swapaxes(1,2)[1:,1:,:], rank.swapaxes(1,2)[1:,1:,:])
    _check_diagonal_axis_helper(
        touches_protein.swapaxes(0,2)[1:,1:,:], rank.swapaxes(0,2)[1:,1:,:])


def determine_touches_protein(struct, grid, probe_radius):

    n_x_cells, n_y_cells, n_z_cells = grid.shape[:3]
    x_min, y_min, z_min = grid[0,0,0]
    x_max, y_max, z_max = grid[-1,-1,-1]

    # determine whether each cell touches protein
    touches_protein = np.zeros(
        (n_x_cells, n_y_cells, n_z_cells), dtype=bool)
    radii = np.array(
        [a.element.radius for a in struct.top.atoms])
    for i in range(struct.top.n_atoms):
        coord = struct.xyz[0, i]
        distance_cutoff = probe_radius + radii[i]
        # get sub-grid of cells that could be within cutoff distance of atom
        min_x_cell_ind, max_x_cell_ind, \
        min_y_cell_ind, max_y_cell_ind, \
        min_z_cell_ind, max_z_cell_ind = _get_cell_inds_within_cutoff(
            grid, coord, distance_cutoff)
        subgrid = grid[
            min_x_cell_ind:max_x_cell_ind+1,
            min_y_cell_ind:max_y_cell_ind+1,
            min_z_cell_ind:max_z_cell_ind+1]

        # find cells in subgrid that are actually within distance cutoff
        # and indicate they are not pockets and that they do touch protein
        offset = np.subtract(subgrid, coord)
        offset_sq = np.einsum('ijkl,ijkl->ijk', offset, offset)
        subgrid_inds_within_cutoff = np.where(offset_sq < (distance_cutoff**2))
        # convert to grid inds
        grid_inds_within_cutoff = (
            subgrid_inds_within_cutoff[0]+min_x_cell_ind,
            subgrid_inds_within_cutoff[1]+min_y_cell_ind,
            subgrid_inds_within_cutoff[2]+min_z_cell_ind)
        touches_protein[grid_inds_within_cutoff] = True
    return touches_protein


def get_pocket_cells(
        struct, grid_spacing=0.1, probe_radius=0.07,
        min_rank=3):
    """Places on a grid on a single structure and identifies all the cells that
    are part of a pocket.

    The algorithm lays a grid over the protein. All cells within the
    distance_cutoff of protein atoms (probe radius + atomic VDW)  are
    discarded as they are assumed to be filled with protein cannot be
    pockets and, therefore, not pockets. Each remaining cell is ranked
    by how many scans through it hit protein on both sides. Seven scans
    are performed, one along each of the x/y/z axes and one along each
    of the four diagonals cutting across a cube centered at the given
    cell.

    Parameters
    ----------
    struct : mdtraj.Tractory
        Only uses the first frame.
    grid_spacing : float, default=0.1 (nm)
        The length of each edge of a cell in nm.  So a cell has a volume of
        grid_spacing^3.
    probe_radius : float, default = 0.07 (nm)
        Radius of pocket cells. Cells within this distance and atomic radius
        (in nm) of a protein atom cannot be pockets.
        The default comes from half the radius of water (half of 0.14 nm).
    min_rank : int, default=3
        The minimum rank a cell has to have to be considered part of a pocket.

    Returns
    -------
    pocket_cells : np.ndarray, shape=(n_cells)
        The x,y,z coordinates of all the cells in teh grid that appear to be
        part of a pocket.
    """

    grid = create_grid(struct, grid_spacing)
    n_x_cells, n_y_cells, n_z_cells = grid.shape[:3]
    x_min, y_min, z_min = grid[0,0,0]
    x_max, y_max, z_max = grid[-1,-1,-1]

    # determine whether each cell touches protein
    touches_protein = determine_touches_protein(struct, grid, probe_radius)

    # rank each remaining pocket cell based on the number of scans that
    # pass through protein on each side
    rank = np.zeros(touches_protein.shape)

    # check along x axis
    _check_cartesian_axis(touches_protein, rank)
    # check along y axis, using views to swap x/y axes
    _check_cartesian_axis(touches_protein.swapaxes(0,1), rank.swapaxes(0,1))
    # check along z axis, using views to swap x/z axes
    _check_cartesian_axis(touches_protein.swapaxes(0,2), rank.swapaxes(0,2))

    # for each of 4 diagonals diagonals, need scan out from 3 faces of
    # grid along pos x, pos y, pos z
    _check_diagonal_axis(touches_protein, rank)
    # along neg x, pos y, pos z
    _check_diagonal_axis(touches_protein[::-1,:,:], rank[::-1,:,:])
    # along neg x, neg y, pos z
    _check_diagonal_axis(touches_protein[::-1,::-1,:], rank[::-1,::-1,:])
    # along pos x, neg y, pos z
    _check_diagonal_axis(touches_protein[:,::-1,:], rank[:,::-1,:])

    pocket_inds = np.where(rank>=min_rank)
    pocket_cells = grid[pocket_inds]

    return pocket_cells


def cluster_pocket_cells(pocket_cells, grid_spacing=0.1, min_cluster_size=0):
    """Identify sets of pocket cells that, together, comprise a single
    contiguous pocket.

    Parameters
    ----------
    pocket_cells : np.ndarray, shape=(n_cells)
        The x,y,z coordinates of all the cells in teh grid that appear to be
        part of a pocket.
    grid_spacing : float, default=0.1 (nm)
        The length of each edge of a cell in nm.  So a cell has a volume of
        grid_spacing^3.
    min_cluster_size : int, default=0
        The minimum number of contiguous pocket cells required to constitute
        a pocket.

    Returns
    -------
    sorted_pockets : np.ndarray, shape=(n_cells,3)
        The same x,y,z coordinates specified in the pocket_cells input but
        reordered to match the sorted_cluster_mapping output.
    sorted_cluster_mapping : np.ndarray, shape=(n_cells)
        Integers (0, 1, 2...) specifying which pocket each of the input pocket
        cells belongs to.
    """

    # cluster into contiguous pockets by merging two cells if they are
    # neighbors use a cutoff distance between grid_spacing and
    # 2*grid_spacing to ensure pocket cells are contiguous
    orig_cluster_mapping = scipy.cluster.hierarchy.fclusterdata(
        pocket_cells, t=grid_spacing*1.5, criterion='distance')

    # make sure numbered from 0, since seem to be numbered from 1
    if orig_cluster_mapping.min() > 0:
        orig_cluster_mapping -= orig_cluster_mapping.min()

    # determine how many pocket cells are in each cluster
    orig_n_clusters = orig_cluster_mapping.max()+1
    num_cells_in_pocket = np.zeros(orig_n_clusters)
    for i in orig_cluster_mapping:
        num_cells_in_pocket[i] += 1

    # renumber so the clusters are ordered from largest to smallest
    sorted_cluster_ids = np.argsort(-num_cells_in_pocket)
    sorted_cluster_mapping = []
    sorted_pockets = []
    i = 0
    next_largest_cluster_size = num_cells_in_pocket[sorted_cluster_ids[i]]
    while next_largest_cluster_size > min_cluster_size:
        inds_in_cluster = np.where(
            orig_cluster_mapping==sorted_cluster_ids[i])[0]
        for j in inds_in_cluster:
            sorted_cluster_mapping.append(i)
            sorted_pockets.append(pocket_cells[j])
        i += 1
        if i == orig_n_clusters:
            break
        next_largest_cluster_size = num_cells_in_pocket[sorted_cluster_ids[i]]

    sorted_cluster_mapping = np.array(sorted_cluster_mapping, dtype=int)
    sorted_pockets = np.array(sorted_pockets)

    return sorted_pockets, sorted_cluster_mapping


def _get_pockets_helper(
        struct, grid_spacing, probe_radius, min_rank, min_cluster_size):
    pocket_cells = get_pocket_cells(
        struct, grid_spacing=grid_spacing, probe_radius=probe_radius,
        min_rank=min_rank)
    sorted_pockets, sorted_cluster_mapping = cluster_pocket_cells(
        pocket_cells, grid_spacing=grid_spacing,
        min_cluster_size=min_cluster_size)
    pockets_as_mdtraj = xyz_to_mdtraj(
        sorted_pockets, cluster_ids=sorted_cluster_mapping)
    return pockets_as_mdtraj


def get_pockets(
        traj, grid_spacing=0.1, probe_radius=0.07, min_rank=3,
        min_cluster_size=0, n_procs=1):
    """Finds the pockets in each frame of a trajectory.

    The algorithm lays a grid over the protein. All cells within the
    distance_cutoff of protein atoms are discarded as they are assumed to be
    filled with protein cannot be pockets and, therefore, not pockets.
    Each remaining cell is ranked by how many scans through it hit protein
    on both sides. Seven scans are performed, one along each of the x/y/z axes
    and one along each of the four diagonals cutting across a cube centered at
    the given cell.

    Parameters
    ----------
    traj : mdtraj.Tractory
    grid_spacing : float, default=0.1 (nm)
        The length of each edge of a cell in nm.  So a cell has a volume of
        grid_spacing^3.
    probe_radius : float, default = 0.07 (nm)
        Radius of pocket cells. Cells within this distance and atomic radius
        (in nm) of a protein atom cannot be pockets.
        The default comes from half the radius of water (half of 0.14 nm).
    min_rank : int, default=3
        The minimum rank a cell has to have to be considered part of a pocket.
    min_cluster_size : int, default=0
        Only read every stride-th frame.
    n_procs : int, default=1
        Number of processors to use.

    Returns
    -------
    pockets : List of mdtraj.Trajectory objects
        Each element of the list is an mdtraj.Trajectory representing the
        pockets in a frame in the input Trajectory object. All the cells making
        up a pocket are represented with a carbon atom. Contiguous cells that,
        together, make up a pocket are all contained in the same residue. All
        of these residues are given the name POK. The pockets are listed from
        largest to smallest.
    """

    traj_pockets = Parallel(n_jobs=n_procs)(delayed(_get_pockets_helper)(struct, grid_spacing, probe_radius, min_rank, min_cluster_size) for struct in traj)

    return traj_pockets
