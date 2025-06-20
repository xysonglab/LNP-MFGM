import torch
from scipy.spatial.transform import Rotation

import cgflow.util.rdkit as smolRD

_T = torch.Tensor
TupleRot = tuple[float, float, float]

# *************************************************************************************************
# ********************************** Tensor Util Functions ****************************************
# *************************************************************************************************


def pad_and_stack(data, ptr, length=None, device="cuda"):
    """Pad and stack a list of data tensors.
    Args:
        data: Tensor of shape (num_data, dim)
        ptr: Tensor of shape (batch_size + 1) with the pointers to the start of each batch
    """
    dim_size = data.shape[1]
    batch_size = ptr.shape[0] - 1

    if length is None:
        length = torch.max(ptr[1:] - ptr[:-1])

    result = torch.zeros((batch_size, length, dim_size))
    for i in range(batch_size):
        start, end = ptr[i], ptr[i + 1]
        result[i, :end - start, :] = data[start:end]

    return result.to(device)


def pad_tensors(tensors: list[_T], pad_dim: int = 0) -> _T:
    """Pad a list of tensors with zeros

    All dimensions other than pad_dim must have the same shape. A single tensor is returned with the batch dimension
    first, where the batch dimension is the length of the tensors list.

    Args:
        tensors (list[torch.Tensor]): List of tensors
        pad_dim (int): Dimension on tensors to pad. All other dimensions must be the same size.

    Returns:
        torch.Tensor: Batched, padded tensor, if pad_dim is 0 then shape [B, L, *] where L is length of longest tensor.
    """

    if pad_dim != 0:
        # TODO
        raise NotImplementedError()

    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return padded


# TODO replace with tensor version below
def one_hot_encode(indices: list[int], vocab_size: int) -> _T:
    """Create one-hot encodings from a list of indices

    Args:
        indices (list[int]): List of indices into one-hot vectors
        vocab_size (int): Length of returned vectors

    Returns:
        torch.Tensor: One-hot encoded vectors, shape [L, vocab_size] where L is length of indices list
    """

    one_hots = torch.zeros((len(indices), vocab_size), dtype=torch.int64)

    for batch_idx, vocab_idx in enumerate(indices):
        one_hots[batch_idx, vocab_idx] = 1

    return one_hots


# TODO test
def one_hot_encode_tensor(indices: _T, vocab_size: int) -> _T:
    """Create one-hot encodings from indices

    Args:
        indices (torch.Tensor): Indices into one-hot vectors, shape [*, L]
        vocab_size (int): Length of returned vectors

    Returns:
        torch.Tensor: One-hot encoded vectors, shape [*, L, vocab_size]
    """

    one_hot_shape = (*indices.shape, vocab_size)
    one_hots = torch.zeros(one_hot_shape,
                           dtype=torch.int64,
                           device=indices.device)
    one_hots.scatter_(-1, indices.unsqueeze(-1), 1)
    return one_hots


def atomics_to_index(atomics: _T, vocab) -> _T:
    atomic_nums = [int(atomic) for atomic in atomics.tolist()]
    tokens = [smolRD.PT.symbol_from_atomic(atomic) for atomic in atomic_nums]
    atomics_idxs = torch.tensor(
        vocab.indices_from_tokens(tokens, one_hot=False))
    return atomics_idxs


def one_hot_encode_atomics(atomics: _T, vocab) -> _T:
    """Create one-hot encodings for atomic numbers"""
    atomic_nums = [int(atomic) for atomic in atomics.tolist()]
    tokens = [smolRD.PT.symbol_from_atomic(atomic) for atomic in atomic_nums]
    one_hot_atomics = torch.tensor(
        vocab.indices_from_tokens(tokens, one_hot=True))
    return one_hot_atomics


def charge_to_index(charges: _T) -> _T:
    """converts charges to indices"""
    charge_idxs = [
        smolRD.CHARGE_IDX_MAP[charge] for charge in charges.tolist()
    ]
    charge_idxs = torch.tensor(charge_idxs)
    return charge_idxs


def aa_to_index(aas: list) -> _T:
    """converts amino acids to indices"""
    aas = [aa if aa in smolRD.RESIDUE_IDX_MAP else "UNK" for aa in aas]
    aa_idxs = [smolRD.RESIDUE_IDX_MAP[aa] for aa in aas]
    aa_idxs = torch.tensor(aa_idxs)
    return aa_idxs


def pairwise_concat(t: _T) -> _T:
    """Concatenates two representations from all possible pairings in dimension 1

    Computes all possible pairs of indices into dimension 1 and concatenates whatever representation they have in
    higher dimensions. Note that all higher dimensions will be flattened. The output will have its shape for
    dimension 1 duplicated in dimension 2.

    Example:
    Input shape [100, 16, 128]
    Output shape [100, 16, 16, 256]
    """

    idx_pairs = torch.cartesian_prod(*((torch.arange(t.shape[1]), ) * 2))
    output = t[:, idx_pairs].view(t.shape[0], t.shape[1], t.shape[1], -1)
    return output


def segment_sum(data, segment_ids, num_segments):
    """Computes the sum of data elements that are in each segment

    The inputs must have shapes that look like the following:
    data [batch_size, seq_length, num_features]
    segment_ids [batch_size, seq_length], must contain integers

    Then the output will have the following shape:
    output [batch_size, num_segments, num_features]
    """

    err_msg = "data and segment_ids must have the same shape in the first two dimensions"
    assert data.shape[0:2] == segment_ids.shape[0:2], err_msg

    result_shape = (data.shape[0], num_segments, data.shape[2])
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, -1, data.shape[2])
    result.scatter_add_(1, segment_ids, data)
    return result


# *************************************************************************************************
# ******************************* Functions for handling edges ************************************
# *************************************************************************************************


def compute_local_mask_batched(adj, max_hops=2):
    """
    Compute a local mask based on bond connectivity for batched data.

    Args:
        adj (torch.Tensor): Batched adjacency matrix of shape (B, N, N) with 1 for a bond, 0 otherwise.
        max_hops (int): Maximum number of bond hops to consider for local grouping.

    Returns:
        torch.Tensor: Boolean mask of shape (B, N, N) where True indicates that a pair of atoms
                      is connected within max_hops bonds.
    """
    B, N, _ = adj.shape
    # Start with self-connections (0 hops): each atom is connected to itself.
    local_mask = torch.eye(N, dtype=torch.bool,
                           device=adj.device).unsqueeze(0).expand(B, N, N)

    # Direct bonds (1 hop)
    current = (adj > 0).bool()  # (B, N, N)
    local_mask = local_mask | current

    # Expand connectivity for additional hops.
    for _ in range(2, max_hops + 1):
        # Use batched matrix multiplication to propagate connections.
        # Casting to float for multiplication, then thresholding.
        current = torch.matmul(current.float(), adj.float()) > 0
        local_mask = local_mask | current

    return local_mask


def adj_from_node_mask(node_mask, self_connect=False):
    """Creates an edge mask from a given node mask assuming all nodes are fully connected excluding self-connections

    Args:
        node_mask (torch.Tensor): Node mask tensor, shape [batch_size, num_nodes], 1 for real node 0 otherwise
        self_connect (bool): Whether to include self connections in the adjacency

    Returns:
        torch.Tensor: Adjacency tensor, shape [batch_size, num_nodes, num_nodes], 1 for real edge 0 otherwise
    """

    num_nodes = node_mask.size()[1]

    # Matrix mult gives us an outer product on the node mask, which is an edge mask
    mask = node_mask.float()
    adjacency = torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1))
    adjacency = adjacency.long()

    # Set diagonal connections
    node_idxs = torch.arange(num_nodes)
    self_mask = node_mask if self_connect else torch.zeros_like(node_mask)
    adjacency[:, node_idxs, node_idxs] = self_mask

    return adjacency


def _pad_edges(edges, max_edges, value=0):
    """Add fake edges to an edge tensor so that the shape matches max_edges

    Args:
        edges (torch.Tensor): Unbatched edge tensor, shape [num_edges, 2], each element is a node index for the edge
        max_edges (int): The number of edges the output tensor should have
        value (int): Padding value, default 0

    Returns:
        (torch.Tensor, torch.Tensor): Tuple of padded edge tensor and padding mask. Shapes [max_edges, 2] for edge
                tensor and [max_edges] for mask. Mask is one for pad elements, 0 otherwise.
    """

    num_edges = edges.size(0)
    mask_kwargs = {"dtype": torch.int64, "device": edges.device}

    if num_edges > max_edges:
        raise ValueError(
            "Number of edges in edge tensor to be padded cannot be greater than max_edges."
        )

    add_edges = max_edges - num_edges

    if add_edges == 0:
        pad_mask = torch.zeros(num_edges, **mask_kwargs)
        return edges, pad_mask

    pad = (0, 0, 0, add_edges)
    padded = torch.nn.functional.pad(edges, pad, mode="constant", value=value)

    zeros_mask = torch.zeros(num_edges, **mask_kwargs)
    ones_mask = torch.ones(add_edges, **mask_kwargs)
    pad_mask = torch.cat((zeros_mask, ones_mask), dim=0)

    return padded, pad_mask


# TODO change callers to use bonds_from_adj
def edges_from_adj(adj_matrix):
    """Flatten an adjacency matrix into a 1D edge representation

    Args:
        adj_matrix (torch.Tensor): Batched adjacency matrix, shape [batch_size, num_nodes, num_nodes]. It can contain
                any non-zero integer for connected nodes but must be 0 for unconnected nodes.

    Returns:
        A tuple of the edge tensor and the edge mask tensor. The edge tensor has shape [batch_size, max_num_edges, 2]
        and the mask [batch_size, max_num_edges]. The mask contains 1 for real edges, 0 otherwise.
    """

    adj_ones = torch.zeros_like(adj_matrix).int()
    adj_ones[adj_matrix != 0] = 1

    # Pad each batch element by a seperate amount so that they can all be packed into a tensor
    # It might be possible to do this in batch form without iterating, but for now this will do
    num_edges = adj_ones.sum(dim=(1, 2)).tolist()
    edge_tuples = list(adj_matrix.nonzero()[:, 1:].split(num_edges))
    padded = [
        _pad_edges(edges, max(num_edges), value=0) for edges in edge_tuples
    ]

    # Unravel the padded tuples and stack them into batches
    edge_tuples_padded, pad_masks = tuple(zip(*padded, strict=False))
    edges = torch.stack(edge_tuples_padded).long()
    edges = (edges[:, :, 0], edges[:, :, 1])
    edge_mask = (torch.stack(pad_masks) == 0).long()
    return edges, edge_mask


# TODO test and merge with edges_from_adj
def bonds_from_adj(adj_matrix, lower_tri=True):
    """Flatten an adjacency matrix into a 1D edge representation

    Args:
        adj_matrix (torch.Tensor): Adjacency matrix, can be batched or not, shape [batch_size, num_nodes, num_nodes].
            Each item in the matrix corrsponds to the bond type and will be placed into index 2 on dim 1 in bonds.
        lower_tri (bool): Whether to only consider bonds which sit in the lower triangular of adj_matrix.

    Returns:
        An bond list tensor, shape [batch_size, num_bonds, 3]. If an item is a padding bond index 2 on the last
            dimension will be 0.
    """

    batched = True
    if len(adj_matrix.shape) == 2:
        adj_matrix = adj_matrix.unsqueeze(0)
        batched = False

    if lower_tri:
        adj_matrix = torch.tril(adj_matrix, diagonal=-1)

    bonds = []
    for adj in list(adj_matrix):
        bond_indices = adj.nonzero()
        bond_types = adj[bond_indices[:, 0], bond_indices[:, 1]]
        bond_list = torch.cat((bond_indices, bond_types.unsqueeze(-1)), dim=-1)
        bonds.append(bond_list)

    # Bonds will be padded with 0s so the bond type will tell whether the bond is real or not
    bonds = pad_tensors(bonds, pad_dim=0)
    if not batched:
        bonds = bonds.squeeze(0)

    return bonds


def adj_from_edges(edge_indices: _T,
                   edge_types: _T,
                   n_nodes: int,
                   symmetric: bool = False):
    """Create adjacency matrix from a list of edge indices and types

    If an edge pair appears multiple times with different edge types, the adj element for that edge is undefined.

    Args:
        edge_indices (torch.Tensor): Edge list tensor, shape [n_edges, 2]. Pairs of (from_idx, to_idx).
        edge_types (torch.Tensor): Edge types, shape either [n_edges] or [n_edges, edge_types].
        n_nodes (int): Number of nodes in the adjacency matrix. This must be >= to the max node index in edges.
        symmetric (bool): Whether edges are considered symmetric. If True the adjacency matrix will also be symmetric,
                otherwise only the exact node indices within edges will be used to create the adjacency.

    Returns:
        torch.Tensor: Adjacency matrix tensor, shape [n_nodes, n_nodes] or
                [n_nodes, n_nodes, edge_types] if distributions over edge types are provided.
    """

    assert len(edge_indices.shape) == 2
    assert edge_indices.shape[0] == edge_types.shape[0]
    assert edge_indices.size(1) == 2

    adj_dist = len(edge_types.shape) == 2

    edge_indices = edge_indices.long()
    edge_types = edge_types.float() if adj_dist else edge_types.long()

    if adj_dist:
        shape = (n_nodes, n_nodes, edge_types.size(-1))
        adj = torch.zeros(shape, device=edge_indices.device, dtype=torch.float)

    else:
        shape = (n_nodes, n_nodes)
        adj = torch.zeros(shape, device=edge_indices.device, dtype=torch.long)

    from_indices = edge_indices[:, 0]
    to_indices = edge_indices[:, 1]

    adj[from_indices, to_indices] = edge_types
    if symmetric:
        adj[to_indices, from_indices] = edge_types

    return adj


def edges_from_nodes(coords,
                     k=None,
                     node_mask=None,
                     edge_format="adjacency",
                     self_connect=False):
    """Constuct edges from node coords

    Connects a node to its k nearest nodes. If k is None then connects each node to all its neighbours. A node is
    never connected to itself.

    Args:
        coords (torch.Tensor): Node coords, shape [batch_size, num_nodes, 3]
        k (int): Number of neighbours to connect each node to, None means connect to all nodes except itself
        node_mask (torch.Tensor): Node mask, shape [batch_size, num_nodes], 1 for real nodes 0 otherwise
        edge_format (str): Edge format, should be either 'adjacency' or 'list'
        self_connect (bool): Whether to connect each node to itself

    Returns:
        If format is 'adjacency' this returns an adjacency matrix, shape [batch_size, num_nodes, num_nodes] which
        contains 1 for connected nodes and 0 otherwise. Note that if a value for k is provided the adjacency matrix
        may not be symmetric and should always be used s.t. 'from nodes' are in dim 1 and 'to nodes' are in dim 2.

        If format is 'list' this returns the tuple (edges, edge mask), edges is also a two-tuple of tensors, each of
        shape [batch_size, num_edges], specifying node indices for each edge. The edge mask has shape
        [batch_size, num_edges] and contains 1 for 'real' edges and 0 otherwise.
    """

    if edge_format not in ["adjacency", "list"]:
        raise ValueError(f"Unrecognised edge format '{edge_format}'")

    adj_format = edge_format == "adjacency"
    batch_size, num_nodes, _ = coords.size()

    # If node mask is None all nodes are real
    if node_mask is None:
        node_mask = torch.ones((batch_size, num_nodes),
                               device=coords.device,
                               dtype=torch.int64)

    adj_matrix = adj_from_node_mask(node_mask)

    if k is not None:
        k = min(k, num_nodes - 1)

        # Find k closest nodes for each node
        dists = calc_distances(coords)
        dists[adj_matrix == 0] = float("inf")

        _, best_idxs = dists.topk(k, dim=2, largest=False)

        # Adjust adj matrix to only have k connections per node
        k_adj_matrix = torch.zeros_like(adj_matrix)
        batch_idxs = torch.arange(batch_size).view(-1, 1,
                                                   1).expand(-1, num_nodes, k)
        node_idxs = torch.arange(num_nodes).view(1, -1,
                                                 1).expand(batch_size, -1, k)
        k_adj_matrix[batch_idxs, node_idxs, best_idxs] = 1

        # Ensure that there are no connections to fake nodes
        k_adj_matrix[adj_matrix == 0] = 0
        adj_matrix = k_adj_matrix

    if self_connect:
        # note adj_matrix is [batch_size, num_nodes, num_nodes]
        # so we cannot just use adj_matrix.fill_diagonal_(1)
        # instead we need to iterate over the batch dimension
        for i in range(batch_size):
            adj_matrix[i].fill_diagonal_(1)

    if adj_format:
        return adj_matrix

    edges, edge_mask = edges_from_adj(adj_matrix)
    return edges, edge_mask


def calc_cross_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Squared‐Euclidean distances between two point clouds.
    Args:
        x: [B, N1, D]
        y: [B, N2, D]
    Returns:
        dists: [B, N1, N2]
    """
    # (x_i - y_j)^2 summed over D
    diffs = x.unsqueeze(2) - y.unsqueeze(1)
    return (diffs * diffs).sum(-1)


def edges_from_two_sets(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    k: int | None = None,
    node_mask1: torch.Tensor | None = None,
    node_mask2: torch.Tensor | None = None,
    edge_format: str = "adjacency",
):
    """
    Build bipartite edges from set1→set2.

    Args:
      coords1: [B, N1, 3]
      coords2: [B, N2, 3]
      k:       how many neighbors in set2 for each node in set1 (None ⇒ all)
      node_mask1: [B, N1] 1=real node, 0=pad
      node_mask2: [B, N2]
      edge_format: "adjacency" or "list"
    Returns:
      If adjacency:
        adj: [B, N1, N2]  {0,1} mask
      If list:
        ( (idx1, idx2), edge_mask )
          idx1,idx2: [B, num_edges]  row/col indices
          edge_mask: [B, num_edges]  1=real edge, 0=pad
    """
    if edge_format not in ("adjacency", "list"):
        raise ValueError(f"Bad format {edge_format!r}")

    B, N1, _ = coords1.shape
    _, N2, _ = coords2.shape

    # default masks = all real
    if node_mask1 is None:
        node_mask1 = coords1.new_ones(B, N1, dtype=torch.int64)
    if node_mask2 is None:
        node_mask2 = coords2.new_ones(B, N2, dtype=torch.int64)

    # initial bipartite mask [B,N1,N2]
    valid = node_mask1[:, :, None] * node_mask2[:, None, :]

    # if k‑nearest
    if k is not None:
        k = min(k, N2)
        
        dists = calc_cross_distances(coords1, coords2)  # [B,N1,N2]
        dists = dists.masked_fill(valid == 0, float("inf"))

        # find the k closest in set2 for each node1
        _, best = dists.topk(k, dim=2, largest=False)  # [B,N1,k]

        # build new adjacency
        adj = torch.zeros_like(valid)
        batch_idx = torch.arange(B, device=coords1.device)[:, None,
                                                           None].expand(
                                                               -1, N1, k)
        row_idx = torch.arange(N1,
                               device=coords1.device)[None, :,
                                                      None].expand(B, -1, k)

        adj[batch_idx, row_idx, best] = 1
        adj = adj * valid  # ban any edges to pad‐nodes
    else:
        adj = valid

    if edge_format == "adjacency":
        return adj

    # --- now build the “list” form
    if k is not None:
        # fixed N1*k edges per batch
        E = N1 * k
        # flatten the same row_idx and best arrays we used above:
        idx1 = row_idx.reshape(B, E)
        idx2 = best.reshape(B, E)
        mask = adj[batch_idx, row_idx, best].reshape(B, E)
        return (idx1, idx2), mask

    else:
        # dynamic number per batch: fall back to gathering nonzeros per‐batch
        edge_idxs = []
        edge_masks = []
        for b in range(B):
            i, j = torch.nonzero(adj[b], as_tuple=True)
            edge_idxs.append((i, j))
            edge_masks.append(torch.ones_like(i, dtype=torch.int64))
        return edge_idxs, edge_masks


def gather_edge_features(pairwise_feats, adj_matrix):
    """Gather edge features for each node from pairwise features using the adjacency matrix

    All 'from nodes' (dimension 1 on the adj matrix) must have the same number of edges to 'to nodes'. Practically
    this means that the number of non-zero elements in dimension 2 of the adjacency matrix must always be the same.

    Args:
        pairwise_feats (torch.Tensor): Pairwise features tensor, shape [batch_size, num_nodes, num_nodes, num_feats]
        adj_matrix (torch.Tensor): Batched adjacency matrix, shape [batch_size, num_nodes, num_nodes]. It can contain
                any non-zero integer for connected nodes but must be 0 for unconnected nodes.

    Returns:
        torch.Tensor: Dense feature matrix, shape [batch_size, num_nodes, edges_per_node, num_feats]
    """

    # In case some of the connections don't use 1, create a 1s adjacency matrix
    adj_ones = torch.zeros_like(adj_matrix).int()
    adj_ones[adj_matrix != 0] = 1

    num_neighbours = adj_ones.sum(dim=2)
    feats_per_node = num_neighbours[0, 0].item()

    assert (num_neighbours == feats_per_node
            ).all(), "All nodes must have the same number of connections"

    if len(pairwise_feats.size()) == 3:
        batch_size, num_nodes, _ = pairwise_feats.size()
        pairwise_feats = pairwise_feats.unsqueeze(3)

    elif len(pairwise_feats.size()) == 4:
        batch_size, num_nodes, _, _ = pairwise_feats.size()

    # nonzero() orders indices lexicographically with the last index changing the fastest, so we can reshape the
    # indices into a dense form with nodes along the outer axis and features along the inner
    gather_idxs = adj_ones.nonzero()[:, 2].reshape(
        (batch_size, num_nodes, feats_per_node))
    batch_idxs = torch.arange(batch_size).view(-1, 1, 1)
    node_idxs = torch.arange(num_nodes).view(1, -1, 1)
    dense_feats = pairwise_feats[batch_idxs, node_idxs, gather_idxs, :]
    if dense_feats.size(-1) == 1:
        return dense_feats.squeeze(-1)

    return dense_feats


# *************************************************************************************************
# ********************************* Geometric Util Functions **************************************
# *************************************************************************************************


# TODO rename? Maybe also merge with inter_distances
# TODO test unbatched and coord sets inputs
def calc_distances(coords, edges=None, sqrd=False, eps=1e-6):
    """Computes distances between connected nodes

    Takes an optional edges argument. If edges is None this will calculate distances between all nodes and return the
    distances in a batched square matrix [batch_size, num_nodes, num_nodes]. If edges is provided the distances are
    returned for each edge in a batched 1D format [batch_size, num_edges].

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [batch_size, num_nodes, 3]
        edges (tuple): Two-tuple of connected node indices, each tensor has shape [batch_size, num_edges]
        sqrd (bool): Whether to return the squared distances
        eps (float): Epsilon to add before taking the square root for numical stability in the gradients

    Returns:
        torch.Tensor: Distances tensor, the shape depends on whether edges is provided (see above).
    """

    # TODO add checks

    # Create fake batch dim if unbatched
    unbatched = False
    if len(coords.size()) == 2:
        coords = coords.unsqueeze(0)
        unbatched = True

    if edges is None:
        coord_diffs = coords.unsqueeze(-2) - coords.unsqueeze(-3)
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=-1)

    else:
        edge_is, edge_js = edges
        batch_index = torch.arange(coords.size(0)).unsqueeze(1)
        coord_diffs = coords[batch_index, edge_js, :] - coords[batch_index,
                                                               edge_is, :]
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=2)

    sqrd_dists = sqrd_dists.squeeze(0) if unbatched else sqrd_dists

    if sqrd:
        return sqrd_dists

    return torch.sqrt(sqrd_dists + eps)


def inter_distances(coords1, coords2, sqrd=False, eps=1e-6):
    # TODO add checks and doc

    # Create fake batch dim if unbatched
    unbatched = False
    if len(coords1.size()) == 2:
        coords1 = coords1.unsqueeze(0)
        coords2 = coords2.unsqueeze(0)
        unbatched = True

    coord_diffs = coords1.unsqueeze(2) - coords2.unsqueeze(1)
    sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=3)
    sqrd_dists = sqrd_dists.squeeze(0) if unbatched else sqrd_dists

    if sqrd:
        return sqrd_dists

    return torch.sqrt(sqrd_dists + eps)


def calc_com(coords, node_mask=None):
    """Calculates the centre of mass of a pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM of pointclouds with imaginary nodes excluded, shape [*, 1, 3]
    """

    node_mask = torch.ones_like(coords[...,
                                       0]) if node_mask is None else node_mask

    assert node_mask.shape == coords[..., 0].shape

    num_nodes = node_mask.sum(dim=-1)
    real_coords = coords * node_mask.unsqueeze(-1)
    com = real_coords.sum(dim=-2) / num_nodes.unsqueeze(-1)
    return com.unsqueeze(-2)


def zero_com(coords, node_mask=None):
    """Sets the centre of mass for a batch of pointclouds to zero for each pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM-free coordinates, where imaginary nodes are excluded from CoM calculation
    """

    com = calc_com(coords, node_mask=node_mask)
    shifted = coords - com
    return shifted


def standardise_coords(coords, node_mask=None):
    """Convert coords into a standard normal distribution

    This will first remove the centre of mass from all pointclouds in the batch, then calculate the (biased) variance
    of the shifted coords and use this to produce a standard normal distribution.

    Args:
        coords (torch.Tensor):  Coordinate tensor, shape [batch_size, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [batch_size, num_nodes], 1 for real node, 0 otherwise

    Returns:
        Tuple[torch.Tensor, float]: The standardised coords and the variance of the original coords
    """

    if node_mask is None:
        node_mask = torch.ones_like(coords)[:, :, 0]

    coord_idxs = node_mask.nonzero()
    real_coords = coords[coord_idxs[:, 0], coord_idxs[:, 1], :]

    variance = torch.var(real_coords, correction=0)
    std_dev = torch.sqrt(variance)

    result = (coords / std_dev) * node_mask.unsqueeze(2)
    return result, std_dev.item()


def rotate(coords: torch.Tensor, rotation: Rotation | TupleRot):
    """Rotate coordinates for a single molecule

    Args:
        coords (torch.Tensor): Unbatched coordinate tensor, shape [num_atoms, 3]
        rotation (Union[Rotation, Tuple[float, float, float]]): Can be either a scipy Rotation object or a tuple of
                rotation values in radians, (x, y, z). These are treated as extrinsic rotations. See the scipy docs
                (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) for info.

    Returns:
        torch.Tensor: Rotated coordinates
    """

    if not isinstance(rotation, Rotation):
        rotation = Rotation.from_euler("xyz", rotation)

    device = coords.device
    coords = coords.cpu().numpy()

    rotated = rotation.apply(coords)
    rotated = torch.tensor(rotated, device=device)
    return rotated


def cartesian_to_spherical(coords):
    sqrd_dists = (coords * coords).sum(dim=-1)
    radii = torch.sqrt(sqrd_dists)
    inclination = torch.acos(coords[..., 2] / radii).unsqueeze(2)
    azimuth = torch.atan2(coords[..., 1], coords[..., 0]).unsqueeze(2)
    spherical = torch.cat((radii.unsqueeze(2), inclination, azimuth), dim=-1)
    return spherical


# *************************************************************************************************
# ************************************** Util Classes *********************************************
# *************************************************************************************************


class SparseFeatures:

    def __init__(self, dense, idxs):
        assert len(dense.size()) == 3
        assert dense.size() == idxs.size()

        batch_size, num_nodes, num_feats = dense.size()

        self.bs = batch_size
        self.num_nodes = num_nodes
        self.num_feats = num_feats

        self._dense = dense
        self._idxs = idxs

    @staticmethod
    def from_sparse(sparse_feats, adj_matrix, feats_per_node):
        err_msg = "adj_matrix must have feats_per_node ones in each row"
        assert sparse_feats.size() == adj_matrix.size(
        ), "sparse_feats and adj_matrix must have the same shape"
        assert adj_matrix.size()[1] == adj_matrix.size(
        )[2], "adj_matrix must be square"
        assert (adj_matrix.sum(dim=2) == feats_per_node).all().item(), err_msg

        batch_size, num_nodes, _ = adj_matrix.size()
        feat_idxs = adj_matrix.nonzero()[:, 2].reshape(
            (batch_size, num_nodes, feats_per_node))
        dense_feats = torch.gather(sparse_feats, 2, feat_idxs)
        return SparseFeatures(dense_feats, feat_idxs)

    @staticmethod
    def from_dense(dense_feats, idxs):
        return SparseFeatures(dense_feats, idxs)

    def to_tensor(self):
        sparse_matrix = torch.zeros((self.bs, self.num_nodes, self.num_nodes),
                                    device=self._dense.device)
        sparse_matrix.scatter_(2, self._idxs, self._dense)
        return sparse_matrix

    def mult(self, other):
        if isinstance(other, int | float):
            return self.from_dense(self._dense * other, self._idxs)

        if not torch.is_tensor(other):
            raise TypeError(
                "Object to multiply by must be an int, float or torch.Tensor")

        assert other.size() == (self.bs, self.num_nodes, self.num_nodes)

        other_dense = torch.gather(other, 2, self._idxs)
        return self.from_dense(self._dense * other_dense, self._idxs)

    def matmul(self, other):
        if not torch.is_tensor(other):
            raise TypeError("Object to multiply by must be a torch.Tensor")

        assert tuple(other.size()[:2]) == (self.bs, self.num_nodes)

        # There doesn't seem to be an efficient implementation of sparse batched matmul available atm, so just do
        # regular matmul instead. We will still get some speed benefit from having lots of zeros.
        tensor = self.to_tensor()
        return torch.bmm(tensor, other)

    def softmax(self):
        dense_softmax = torch.softmax(self._dense, dim=2)
        return self.from_dense(dense_softmax, self._idxs)

    def dropout(self, p, train=False):
        dense_dropout = torch.dropout(self._dense, p, train=train)
        return self.from_dense(dense_dropout, self._idxs)

    def add(self, other):
        """Add a matrix only at elements which are not sparse in self"""

        assert len(other.size()) == 3

        other_dense = torch.gather(other, 2, self._idxs)
        return self.from_dense(self._dense + other_dense, self._idxs)

    def sum(self, dim=None):
        if dim == 1:
            return self.to_tensor().sum(dim=1)

        return self._dense.sum(dim=dim)
