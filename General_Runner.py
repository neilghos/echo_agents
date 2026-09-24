import multiprocessing as mp
import pickle
import random
import time
from math import inf
from typing import List, Optional
import numpy as np
from tqdm import tqdm

from src.detector import neural_ecd_runner
from utils import compute_stats_and_D, read_directed_iGraph_from_file, save_pickle


def _run_with_queue(q, func, args):
    try:
        res = func(*args)
        q.put({"status": "ok", "result": res})
    except MemoryError:
        q.put({"status": "out_of_memory"})
    except Exception as e:
        q.put({"status": "error", "error": str(e)})


def isolated_run(func, args, timeout_sec=12 * 60 * 60):
    q = mp.Queue()
    p = mp.Process(target=_run_with_queue, args=(q, func, args))
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
        p.terminate()
        return {"status": "time_limit"}

    if q.empty():
        return {"status": "killed_or_oom"}

    return q.get()


def tester(
    test_id: int,
    dataset_name: str,
    suffixes: Optional[List[str]] = None,
    pre_path: str = "EchoChambersDatasets/",
    seed_id: int = 42,
):
    if suffixes is None:
        suffixes = ["_fj", "_filtering", "_extremes"]

    if "SBM" in dataset_name:
        dataset_name_plus = f"{test_id}_{dataset_name}"
        thresholds = {"theta_V": 0.05, "theta_E": 0.3, "theta_R": 2.0}
    else:
        dataset_name_plus = f"{dataset_name}"
        thresholds = {"theta_V": 0.075, "theta_E": 0.3, "theta_R": 2.0}

    results = {}
    for suffix in suffixes:
        dataset_name_prime = f"{test_id}_{dataset_name}{suffix}"

        result = isolated_run(
            neural_ecd_runner,
            (
                seed_id + test_id,
                dataset_name_plus,
                dataset_name_prime,
                pre_path,
                thresholds,
            ),
            timeout_sec=12 * 60 * 60,
        )

        results[suffix] = result

    return results


def main(
    datasets: List[str],
    test_ids: List[int],
    suffixes: Optional[List[str]] = None,
    pre_path_results: str = "EchoChambersResultsMain/",
    pre_path: str = "EchoChambersDatasets/",
):
    if suffixes is None:
        suffixes = ["_fj", "_filtering", "_extremes"]

    dataset_pbar = tqdm(datasets, desc="Datasets")
    for dataset_name in dataset_pbar:
        dataset_pbar.set_description(f"Dataset: {dataset_name}")
        all_results = {"OurModel": []}

        run_pbar = tqdm(test_ids, desc=f"Seeds for {dataset_name}", leave=False)
        for test_id in run_pbar:
            run_pbar.set_postfix(seed=test_id)
            x = tester(
                test_id=test_id,
                dataset_name=dataset_name,
                suffixes=suffixes,
                pre_path=pre_path,
            )
            all_results["OurModel"].append([dataset_name, test_id, x.copy()])

            # Print brief summary of run results
            for sfx in suffixes:
                sub_res = x.get(sfx, {})
                if sub_res.get("status") == "ok":
                    r = sub_res["result"]
                    r_val = f"{r.get('R', inf):.4f}" if r.get('R', inf) < inf else "inf"
                    print(f"  [{dataset_name} | run {test_id} | {sfx[1:]}] R = {r_val}, size = {r.get('LenEC', 0)}, time = {r.get('time', 0)}s")

        save_pickle(
            pre_path_results,
            name=f"MainExperiments-{dataset_name}.pkl",
            obj=all_results,
        )