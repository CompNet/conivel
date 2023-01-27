import argparse
import glob, json, os, re, shutil
from typing import Optional


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action="store_true")
    args = parser.parse_args()

    run_dirs = [
        d for d in glob.glob(f"runs/*") if re.match(r"[0-9]+", os.path.basename(d))
    ]

    for run_dir in run_dirs:

        with open(f"{run_dir}/config.json") as f:
            config = json.load(f)

        with open(f"{run_dir}/run.json") as f:
            run = json.load(f)

        if run["status"] == "FAILED":
            print(f"[warning] run {run_dir} failed - ignoring...")
            continue

        run_type = run["experiment"]["name"]
        if run_type == "xp_kfolds":
            name = config["context_retriever"]
        elif run_type == "xp_kfolds_neural":
            name = "neural_" + config["retrieval_heuristic"]
        elif run_type == "xp_neural_context_retriever":
            name = "neural_test_" + config["retrieval_heuristic"]
        elif run_type == "xp_ideal_neural_retriever":
            name = "neural_ideal_" + config["retrieval_heuristic"]
        else:
            print(f"[warning] unknown run_type {run_type} - ignoring...")
            continue

        print(f"moving {run_dir} to ./runs/{name}")
        if not args.dry_run:
            shutil.move(run_dir, f"./runs/{name}")
