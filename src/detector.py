import random
import time
from math import inf
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import igraph as ig

from utils import compute_stats_and_D, read_directed_iGraph_from_file
from src.data import extract_seed_subgraph, prepare_graph_data, select_candidate_seeds
from src.models import FeatureGatedMLP
from src.trainer import train_echo_chamber_model
from src.losses import EchoChamberLoss
from tqdm import tqdm


def sweep_cut(
    G: ig.Graph,
    candidate_nodes: List[int],
    scores: np.ndarray,
    opinions_dict: Dict[int, float],
    thresholds: Dict[str, float],
    min_size: int = 10,
    max_size: int = 250,
) -> Dict[str, Union[float, List[int], int]]:
    """
    Performs a sweep cut over candidate nodes ordered by model predicted scores (descending).
    Evaluates candidate prefix sets using the official compute_stats_and_D metric and
    returns the best valid echo chamber with the lowest structural isolation R.
    """
    order = np.argsort(scores)[::-1]
    sorted_nodes = [candidate_nodes[idx] for idx in order]

    best = {
        "R": inf,
        "EC": [],
        "LenEC": 0,
        "V_S": -1.0,
        "Delta_E": -1.0,
    }

    max_k = min(len(sorted_nodes), max_size)
    for k in range(min_size, max_k + 1):
        prefix_set = sorted_nodes[:k]
        E_S, V_S, delta_E, R = compute_stats_and_D(G, prefix_set, opinions_dict)

        if (
            len(prefix_set) >= min_size
            and V_S <= thresholds["theta_V"]
            and delta_E >= thresholds["theta_E"]
            and R < best["R"]
        ):
            best = {
                "R": R,
                "EC": prefix_set.copy(),
                "LenEC": len(prefix_set),
                "V_S": V_S,
                "Delta_E": delta_E,
            }

    return best


def detect_echo_chamber(
    G: ig.Graph,
    opinions: Union[Dict[int, float], List[float]],
    thresholds: Dict[str, float],
    random_seed: int = 42,
    model: Optional[torch.nn.Module] = None,
    device: Optional[str] = None,
    max_seeds: int = 20,
    epochs_per_seed: int = 200,
    hidden_dim: int = 64,
) -> Dict[str, Union[float, List[int], int, str]]:
    """
    Main detection pipeline using FeatureGatedMLP + EchoChamberLoss:
    1. Selects candidate seeds with high extremism and local homophily.
    2. Extracts local ego-subgraphs around promising seeds.
    3. Self-supervised optimization of FeatureGatedMLP to minimize R.
    4. Performs sweep-cut discretization against official ECD thresholds.
    5. Returns the best echo chamber candidate with minimal structural isolation R.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(opinions, list):
        opinions_dict = {i: opinions[i] for i in range(len(opinions))}
    else:
        opinions_dict = opinions

    start_time = time.perf_counter()

    # Step 1: Select candidate seeds
    seeds = select_candidate_seeds(
        G,
        opinions_dict,
        seed_ratio=0.005 if G.vcount() < 50000 else 0.001,
        min_extremism=thresholds["theta_E"],
        max_seeds=max_seeds,
    )

    best_overall = {
        "R": inf,
        "EC": [],
        "LenEC": 0,
        "V_S": -1.0,
        "Delta_E": -1.0,
    }

    loss_fn = EchoChamberLoss().to(device)

    # Step 2: Iterate over promising seeds
    seed_pbar = tqdm(seeds, desc="Evaluating candidate seeds", leave=False)
    for seed in seed_pbar:
        sub_nodes = extract_seed_subgraph(
            G,
            opinions_dict,
            seed_node=seed,
            max_hops=2,
            max_nodes=300,
        )

        if len(sub_nodes) < 10:
            continue

        # Prepare local subgraph tensors
        data = prepare_graph_data(G, opinions_dict, subgraph_nodes=sub_nodes, device=device)

        # Step 3: Train / Infer FeatureGatedMLP on this local subgraph
        if model is None:
            local_model = FeatureGatedMLP(in_features=7, hidden_dim=hidden_dim).to(device)
            local_model, _ = train_echo_chamber_model(
                model=local_model,
                data=data,
                loss_fn=loss_fn,
                num_epochs=epochs_per_seed,
                lr=0.015,
                theta_V=thresholds["theta_V"],
                theta_E=thresholds["theta_E"],
                verbose=False,
            )
        else:
            local_model = model

        with torch.no_grad():
            scores = local_model(data).detach().cpu().numpy().flatten()

        # Step 4: Sweep cut to find optimal prefix
        chamber = sweep_cut(
            G,
            candidate_nodes=sub_nodes,
            scores=scores,
            opinions_dict=opinions_dict,
            thresholds=thresholds,
            min_size=10,
        )

        if chamber["R"] < best_overall["R"]:
            best_overall = chamber.copy()
            print(f"    -> [Seed {seed}] New Best EC! R = {best_overall['R']:.4f} (size={best_overall['LenEC']}, Var={best_overall['V_S']:.4f})")

        r_disp = f"{best_overall['R']:.4f}" if best_overall['R'] < inf else "inf"
        seed_pbar.set_postfix(best_R=r_disp, size=best_overall['LenEC'])

    elapsed = time.perf_counter() - start_time
    best_overall["time"] = f"{elapsed:.2f}"
    return best_overall


def neural_ecd_runner(
    seed_id: int,
    dataset_name: str,
    dataset_name_prime: str,
    pre_path: str,
    thresholds: Dict[str, float],
    model=None,
    device: Optional[str] = None,
) -> Dict[str, Union[float, List[int], int, str]]:
    """
    Adapter function for General_Runner's isolated_run.
    Loads graph and opinions from files and executes detect_echo_chamber.
    """
    G, _ = read_directed_iGraph_from_file(
        filepath=f"{pre_path}{dataset_name}_edges.txt"
    )
    import pickle
    opinions = pickle.load(open(f"{pre_path}{dataset_name_prime}.pkl", "rb"))

    return detect_echo_chamber(
        G=G,
        opinions=opinions,
        thresholds=thresholds,
        random_seed=seed_id,
        model=model,
        device=device,
    )
