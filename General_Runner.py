import multiprocessing as mp
import pickle
import random
import time
from math import inf
import numpy as np

from JECHO import JECHO_main, JECHO_main2
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
    ECD_model,
    test_id,
    dataset_name,
    score_methods=None,
    expansion_methods=None,
    suffixes=None,
    pre_path="EchoChambersDatasets/",
    seed_ratio=None,
):
    seed_id = 42

    if "SBM" in dataset_name:
        dataset_name_plus = f"{test_id}_{dataset_name}"
        thresholds = {"theta_V": 0.05, "theta_E": 0.3, "theta_R": 2}
    else:
        dataset_name_plus = f"{dataset_name}"
        thresholds = {"theta_V": 0.075, "theta_E": 0.3, "theta_R": 2}

    results = {}
    for suffix in suffixes:
        dataset_name_prime = f"{test_id}_{dataset_name}{suffix}"

        if ECD_model == "JECHO":
            result = isolated_run(
                JECHO_main,
                (
                    dataset_name_plus,
                    dataset_name_prime,
                    score_methods,
                    expansion_methods,
                    pre_path,
                    thresholds,
                    seed_id + test_id,
                    seed_ratio,
                ),
                timeout_sec=12 * 60 * 60,
            )
        elif ECD_model in ["OurModel", "NeuralECD"]:
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
        else:
            raise ValueError(f"{ECD_model} is not defined.")

        results[(suffix, ECD_model)] = result

    return results


def tester2(
    ECD_model,
    test_id,
    dataset_name,
    seed_ratios,
    score_methods=None,
    expansion_methods=None,
    suffixes=None,
    pre_path="EchoChambersDatasets/",
):
    seed_id = 42
    if "SBM" in dataset_name:
        dataset_name_plus = f"{test_id}_{dataset_name}"
        thresholds = {"theta_V": 0.05, "theta_E": 0.3, "theta_R": 2}
    else:
        dataset_name_plus = f"{dataset_name}"
        thresholds = {"theta_V": 0.075, "theta_E": 0.3, "theta_R": 2}

    results = {}
    for suffix in suffixes:
        dataset_name_prime = f"{test_id}_{dataset_name}{suffix}"

        if ECD_model == "JECHO":
            result = JECHO_main2(
                dataset_name_plus,
                dataset_name_prime,
                score_methods,
                expansion_methods,
                pre_path,
                thresholds,
                seed_ratios,
                random_seed=seed_id + test_id,
            )
        else:
            raise ValueError(f"{ECD_model} is not defined.")

        results[(suffix, ECD_model)] = result

    return results


def main(
    datasets,
    test_ids,
    ECD_models,
    score_methods=None,
    expansion_methods=None,
    suffixes=None,
    pre_path_results="EchoChambersResultsMain/",
    pre_path="EchoChambersDatasets/",
    seed_ratio=None,
):
    for dataset_name in datasets:
        all_results = {model: [] for model in ECD_models}
        for test_id in test_ids:
            for ECD_model in ECD_models:
                x = tester(
                    ECD_model,
                    test_id,
                    dataset_name,
                    score_methods,
                    expansion_methods,
                    suffixes,
                    pre_path,
                    seed_ratio,
                )
                all_results[ECD_model].append([dataset_name, test_id, x.copy()])
        save_pickle(
            pre_path_results,
            name=f"MainExperiments-{dataset_name}.pkl",
            obj=all_results,
        )


def main2(
    datasets,
    test_ids,
    ECD_models,
    seed_ratios,
    score_methods=None,
    expansion_methods=None,
    suffixes=None,
    pre_path_results="EchoChambersResultsABL/",
    pre_path="EchoChambersDatasets/",
):
    for dataset_name in datasets:
        all_results = {model: [] for model in ECD_models}
        for test_id in test_ids:
            for ECD_model in ECD_models:
                x = tester2(
                    ECD_model,
                    test_id,
                    dataset_name,
                    seed_ratios,
                    score_methods,
                    expansion_methods,
                    suffixes,
                    pre_path,
                )
                all_results[ECD_model].append([dataset_name, test_id, x.copy()])
        save_pickle(
            pre_path_results,
            name=f"MainExperiments-{dataset_name}.pkl",
            obj=all_results,
        )