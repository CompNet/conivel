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

        is_7b = (
            config["cr_test_dataset_paths"][0]
            == "./runs/gen/genv3/fold0.cr_test_dataset.json"
        )
        size_str = "7b" if is_7b else "13b"

        k = config["cr_heuristics_kwargs"][0]["sents_nb"]

        name = f"neural_book_s{size_str}_k{k}"

        print(f"moving {run_dir} to ./runs/gen/{name}")
        if not args.dry_run:
            shutil.move(run_dir, f"./runs/gen/{name}")
