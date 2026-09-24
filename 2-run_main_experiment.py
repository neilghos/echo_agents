from General_Runner import main as main_runner


if __name__ == "__main__":
    pre_path_results = "EchoChambersResultsMain/"
    pre_path_data = "EchoChambersDatasets/"

    score_methods = ["JiHom"]
    expansion_methods = ["ScoreBasedExpansion", "APPR"]
    suffixes = ["_fj", "_filtering", "_extremes"][:]
    ECD_models = ["OurModel", "JECHO"]

    # 1. Synthetic SBM Experiments (Table 1)
    datasets = [f"SBM_{N}" for N in [5, 10, 50, 100]]
    # main_runner(datasets, list(range(10)), ECD_models, score_methods, expansion_methods, suffixes, pre_path_results, pre_path_data)

    # 2. Real-World Graphs (Table 1)
    # datasets = ["facebook", "lastfm", "git", "twitter"][:]
    # main_runner(datasets, list(range(10)), ECD_models, score_methods, expansion_methods, suffixes, pre_path_results, pre_path_data)

    # 3. Quick test run on SBM_5 (first seed only)
    datasets = ["SBM_5"]
    main_runner(datasets, [0], ECD_models, score_methods, expansion_methods, suffixes, pre_path_results, pre_path_data)