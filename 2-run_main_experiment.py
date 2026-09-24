from General_Runner import main as main_runner


if __name__ == "__main__":
    pre_path_results = "EchoChambersResultsMain/"
    pre_path_data = "EchoChambersDatasets/"
    suffixes = ["_fj", "_filtering", "_extremes"]

    # 1. Quick test run on SBM_5 (seed 0)
    datasets = ["SBM_5"]
    test_ids = [0]

    # 2. To run full synthetic SBM benchmark (Table 1):
    # datasets = [f"SBM_{N}" for N in [5, 10, 50, 100]]
    # test_ids = list(range(10))

    # 3. To run real-world graphs (Table 1):
    # datasets = ["facebook", "lastfm", "git", "twitter"]
    # test_ids = list(range(10))

    main_runner(
        datasets=datasets,
        test_ids=test_ids,
        suffixes=suffixes,
        pre_path_results=pre_path_results,
        pre_path=pre_path_data,
    )