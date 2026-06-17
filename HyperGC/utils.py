import random
from itertools import permutations
from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add
from scipy.sparse import coo_matrix
def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def make_hopmask(hypergraph: dict):
    nodes = set()
    for edge in hypergraph.values():
        nodes.update(edge)
    nodes = sorted(nodes)
    num_nodes = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    inf = float('inf')

    distance_matrix = np.full((num_nodes, num_nodes), inf)
    np.fill_diagonal(distance_matrix, 0)

    for edge in hypergraph.values():
        for i in range(len(edge)):
            for j in range(i + 1, len(edge)):
                idx_i = node_to_idx[edge[i]]
                idx_j = node_to_idx[edge[j]]
                distance_matrix[idx_i, idx_j] = 1
                distance_matrix[idx_j, idx_i] = 1

    for k in tqdm(range(num_nodes)):
        for i in range(num_nodes):
            for j in range(num_nodes):
                distance_matrix[i, j] = min(distance_matrix[i, j], distance_matrix[i, k] + distance_matrix[k, j])
    
    return distance_matrix
def drop_features(x: Tensor, p: float):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x
def drop_features_ptr(masked_nodes, x: Tensor, p: float):
    total=[]
    for i in masked_nodes:
        total+=i
    total=list(set(total))
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    bt=torch.zeros(x.size(0), dtype=torch.bool)
    bt[total]=True

    x = x.clone()
    x[bt][:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]


def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    if p == 0.0:
        return hyperedge_index
    
    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p
    
    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index


def drop_nodes(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_nodes, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[drop_idx, :] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def drop_hyperedges(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float,max_len,seed_select,quan=1.0):
    
    device=hyperedge_index.device
    if p == 0.0:
        return hyperedge_index
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
    Dv = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)  
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    mask_len_edges = (De <= max_len) & (2 <= De)
    drop_mask = torch.rand(num_edges, device=device) < p
    drop_mask = drop_mask & mask_len_edges

    quan=int(torch.quantile(De[drop_mask].float(),quan).item())
    mask_len_edges = (De <= quan) & (2 <= De)
    drop_mask = drop_mask & mask_len_edges
    
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]


    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    masking=H[:,drop_idx]
    H[:, drop_idx] = 0
    hyperedge_index = H.to_sparse().indices()
    
    
    masked_nodes=[]
    start=[]
    edge_size=[]
    if seed_select=="high":
        for col in range(masking.shape[1]):
            nodes_in_edge = torch.where(masking[:, col] == 1)[0]
            if len(nodes_in_edge) == 0:
                continue

            node_degrees = Dv[nodes_in_edge]
            max_idx = torch.argmax(node_degrees)
            seed_node = nodes_in_edge[max_idx]

            start.append(seed_node.item())
            masked_nodes.append(nodes_in_edge.tolist())
            edge_size.append(len(nodes_in_edge))
    elif seed_select=="low":
        for col in range(masking.shape[1]):
            nodes_in_edge = torch.where(masking[:, col] == 1)[0]
            if len(nodes_in_edge) == 0:
                continue

            node_degrees = Dv[nodes_in_edge]
            min_idx = torch.argmin(node_degrees)
            seed_node = nodes_in_edge[min_idx]

            start.append(seed_node.item())
            masked_nodes.append(nodes_in_edge.tolist())
            edge_size.append(len(nodes_in_edge))
    elif seed_select=="random":
        for col in range(masking.shape[1]):
            value = torch.where(masking[:, col] == 1)[0].tolist()

            masked_nodes.append(value)
            start.append(value[0]) #i
            edge_size.append(len(value))
    else:
        AssertionError
    
        
    
    del masking
    del H
    del hyperedge_index
    del drop_mask
    del mask_len_edges
    del De
    del ones
    
    start = torch.tensor(start,dtype=torch.int64).to(device)
    edge_size=torch.tensor(edge_size,dtype=torch.double).to(device)
    
    return masked_nodes,start,edge_size,drop_idx


def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_edges: int):
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def common_node_edge_mask(hyperedge_indexs: list[Tensor], num_nodes: int, num_edges: int):
    hyperedge_weight = hyperedge_indexs[0].new_ones(num_edges)
    node_mask = hyperedge_indexs[0].new_ones((num_nodes,)).to(torch.bool)
    edge_mask = hyperedge_indexs[0].new_ones((num_edges,)).to(torch.bool)

    for index in hyperedge_indexs:
        Dn = scatter_add(hyperedge_weight[index[1]], index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(index.new_ones(index.shape[1]), index[1], dim=0, dim_size=num_edges)
        node_mask &= Dn != 0
        edge_mask &= De != 0
    return node_mask, edge_mask


def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):

    if node_mask is None and edge_mask is None:
        return hyperedge_index

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()
    del H

    return masked_hyperedge_index

def hyperedge_index_masking_batch(hyperedge_index, num_nodes, num_edges, node_mask=None, edge_mask=None):
    device=hyperedge_index.device
    if isinstance(hyperedge_index, torch.Tensor):
        hyperedge_index = hyperedge_index.cpu().numpy()
    if isinstance(node_mask, torch.Tensor):
        node_mask = node_mask.cpu().numpy()
    if isinstance(edge_mask, torch.Tensor):
        edge_mask = edge_mask.cpu().numpy()

    if node_mask is None and edge_mask is None:
        return hyperedge_index 
    
    row, col = hyperedge_index  # Node indices, Hyperedge indices
    data = np.ones(len(row), dtype=np.float32)
    H = coo_matrix((data, (row, col)), shape=(num_nodes, num_edges)).toarray() 

    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = np.vstack(np.nonzero(H[node_mask][:, edge_mask]))
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = np.vstack(np.nonzero(H[:, edge_mask]))
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = np.vstack(np.nonzero(H[node_mask]))
        
    masked_hyperedge_index=torch.from_numpy(masked_hyperedge_index).to(device)
    return masked_hyperedge_index

def clique_expansion(hyperedge_index: Tensor):
    edge_set = set(hyperedge_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)
    
    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(hyperedge_index.device)
