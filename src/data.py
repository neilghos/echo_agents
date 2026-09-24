import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import igraph as ig


@dataclass
class GraphData:
    """Container for graph and opinion tensors ready for neural model execution."""
    x: torch.Tensor                    # (N, feature_dim) node feature matrix
    edge_index: torch.Tensor           # (2, E) directed edge indices
    d_out: torch.Tensor                # (N,) out-degree vector
    d_in: torch.Tensor                 # (N,) in-degree vector
    opinions: torch.Tensor             # (N,) opinion values in [-1, 1]
    global_mean: float                 # Global mean opinion of the full graph
    sparse_adj: torch.Tensor           # (N, N) sparse directed adjacency matrix
    node_map: List[int]                # Mapping from local index to global graph node ID
    num_nodes: int
    num_edges: int

    def to(self, device: Union[str, torch.device]) -> "GraphData":
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.d_out = self.d_out.to(device)
        self.d_in = self.d_in.to(device)
        self.opinions = self.opinions.to(device)
        self.sparse_adj = self.sparse_adj.to(device)
        return self


def compute_node_features(
    opinions_arr: np.ndarray,
    in_degrees: np.ndarray,
    out_degrees: np.ndarray,
    adj_out: List[List[int]],
    global_mean: float,
) -> torch.Tensor:
    """
    Extracts a 7-dimensional node feature vector:
    1. Raw opinion: o_v
    2. Extremism distance: |o_v - global_mean|
    3. In-degree log: log1p(d_in)
    4. Out-degree log: log1p(d_out)
    5. Local neighbor mean opinion
    6. Local neighbor opinion std
    7. Local neighbor agreement ratio (fraction of out-neighbors with same opinion sign)
    """
    n = len(opinions_arr)
    feats = np.zeros((n, 7), dtype=np.float32)

    for i in range(n):
        o_v = opinions_arr[i]
        d_i = in_degrees[i]
        d_o = out_degrees[i]
        nbrs = adj_out[i]

        if len(nbrs) > 0:
            nbr_ops = opinions_arr[nbrs]
            nbr_mean = float(np.mean(nbr_ops))
            nbr_std = float(np.std(nbr_ops))
            same_sign = float(np.mean((nbr_ops * o_v) > 0))
        else:
            nbr_mean = o_v
            nbr_std = 0.0
            same_sign = 1.0

        feats[i, 0] = o_v
        feats[i, 1] = abs(o_v - global_mean)
        feats[i, 2] = math.log1p(d_i)
        feats[i, 3] = math.log1p(d_o)
        feats[i, 4] = nbr_mean
        feats[i, 5] = nbr_std
        feats[i, 6] = same_sign

    return torch.from_numpy(feats)


def prepare_graph_data(
    G: ig.Graph,
    opinions: Union[Dict[int, float], List[float]],
    subgraph_nodes: Optional[List[int]] = None,
    device: str = "cpu",
) -> GraphData:
    """
    Converts an igraph directed graph and opinion mapping into a GraphData object.
    If subgraph_nodes is provided, induces the subgraph over those nodes.
    """
    if isinstance(opinions, list):
        opinions_dict = {i: opinions[i] for i in range(len(opinions))}
    else:
        opinions_dict = opinions

    global_mean = float(np.mean(list(opinions_dict.values())))

    if subgraph_nodes is not None:
        subgraph_nodes = sorted(list(set(subgraph_nodes)))
        node_map = subgraph_nodes
        global_to_local = {g_id: l_id for l_id, g_id in enumerate(subgraph_nodes)}
        n = len(subgraph_nodes)

        sub_edges = []
        for l_u, g_u in enumerate(subgraph_nodes):
            for g_v in G.neighbors(g_u, mode="OUT"):
                if g_v in global_to_local:
                    sub_edges.append((l_u, global_to_local[g_v]))

        opinions_arr = np.array([opinions_dict[g_id] for g_id in subgraph_nodes], dtype=np.float32)
        adj_out = [[] for _ in range(n)]
        # CRITICAL: Use true out-degree in full graph G so that (p @ d_out - E_in) accurately measures real external cut edges
        in_deg = np.array([G.indegree(g_id) for g_id in subgraph_nodes], dtype=np.float32)
        out_deg = np.array([G.outdegree(g_id) for g_id in subgraph_nodes], dtype=np.float32)

        for u, v in sub_edges:
            adj_out[u].append(v)
    else:
        n = G.vcount()
        node_map = list(range(n))
        sub_edges = [(e.source, e.target) for e in G.es]
        opinions_arr = np.array([opinions_dict[i] for i in range(n)], dtype=np.float32)
        in_deg = np.array(G.indegree(), dtype=np.float32)
        out_deg = np.array(G.outdegree(), dtype=np.float32)
        adj_out = [G.neighbors(i, mode="OUT") for i in range(n)]

    x = compute_node_features(opinions_arr, in_deg, out_deg, adj_out, global_mean)

    if len(sub_edges) > 0:
        edge_index = torch.tensor(sub_edges, dtype=torch.long).t().contiguous()
        src, dst = edge_index[0], edge_index[1]
        values = torch.ones(len(src), dtype=torch.float32)
        sparse_adj = torch.sparse_coo_tensor(
            indices=torch.stack([src, dst]),
            values=values,
            size=(n, n),
        ).coalesce()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        sparse_adj = torch.sparse_coo_tensor(size=(n, n)).coalesce()

    data = GraphData(
        x=x,
        edge_index=edge_index,
        d_out=torch.from_numpy(out_deg),
        d_in=torch.from_numpy(in_deg),
        opinions=torch.from_numpy(opinions_arr),
        global_mean=global_mean,
        sparse_adj=sparse_adj,
        node_map=node_map,
        num_nodes=n,
        num_edges=len(sub_edges),
    )
    return data.to(device)


def select_candidate_seeds(
    G: ig.Graph,
    opinions: Union[Dict[int, float], List[float]],
    seed_ratio: float = 0.005,
    min_extremism: float = 0.3,
    max_seeds: int = 50,
) -> List[int]:
    """
    Selects top candidate seed nodes based on:
    1. Extremism: |o_v - global_mean| >= min_extremism
    2. Local opinion homophily: agreement with out-neighbors
    """
    if isinstance(opinions, list):
        opinions_dict = {i: opinions[i] for i in range(len(opinions))}
    else:
        opinions_dict = opinions

    global_mean = float(np.mean(list(opinions_dict.values())))
    n = G.vcount()
    k_seeds = max(10, min(max_seeds, int(math.ceil(n * seed_ratio))))

    candidate_scores = []
    for v in range(n):
        o_v = opinions_dict[v]
        ext = abs(o_v - global_mean)
        if ext < min_extremism:
            continue

        nbrs = G.neighbors(v, mode="OUT")
        if not nbrs:
            continue

        nbr_ops = np.array([opinions_dict[u] for u in nbrs])
        homophily = float(np.mean((nbr_ops * o_v) > 0))
        # Composite score balancing extremism and local agreement
        score = ext * 0.5 + homophily * 0.5
        candidate_scores.append((score, v))

    candidate_scores.sort(key=lambda x: x[0], reverse=True)
    seeds = [v for _, v in candidate_scores[:k_seeds]]
    return seeds


def extract_seed_subgraph(
    G: ig.Graph,
    opinions: Union[Dict[int, float], List[float]],
    seed_node: int,
    max_hops: int = 2,
    max_nodes: int = 500,
) -> List[int]:
    """
    Extracts a local ego-subgraph around seed_node for local neural expansion.
    Only includes nodes with the same opinion polarity (o_v * o_seed > 0)
    to naturally maintain internal opinion homogeneity.
    """
    if isinstance(opinions, list):
        opinions_dict = {i: opinions[i] for i in range(len(opinions))}
    else:
        opinions_dict = opinions

    seed_op = opinions_dict[seed_node]
    subgraph_nodes = {seed_node}
    current_frontier = {seed_node}

    for _ in range(max_hops):
        next_frontier = set()
        for u in current_frontier:
            # Check both outgoing and incoming neighbors in the local community
            nbrs = set(G.neighbors(u, mode="OUT")) | set(G.neighbors(u, mode="IN"))
            for v in nbrs:
                if v not in subgraph_nodes:
                    # Same opinion polarity and proximity check (prevents variance explosion in extremes)
                    if opinions_dict[v] * seed_op > 0 and abs(opinions_dict[v] - seed_op) <= 0.45:
                        next_frontier.add(v)

        subgraph_nodes.update(next_frontier)
        current_frontier = next_frontier

        if len(subgraph_nodes) >= max_nodes:
            break

    # If subgraph has too few nodes, retry with looser distance
    if len(subgraph_nodes) < 15:
        for u in list(subgraph_nodes):
            nbrs = set(G.neighbors(u, mode="OUT")) | set(G.neighbors(u, mode="IN"))
            for v in nbrs:
                if opinions_dict[v] * seed_op > 0:
                    subgraph_nodes.add(v)
                if len(subgraph_nodes) >= max_nodes:
                    break

    # Retain nodes closest in opinion to the seed to keep variance minimal
    if len(subgraph_nodes) > max_nodes:
        sorted_nodes = sorted(
            list(subgraph_nodes),
            key=lambda x: abs(opinions_dict[x] - seed_op),
        )
        return sorted_nodes[:max_nodes]

    return list(subgraph_nodes)
