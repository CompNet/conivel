import argparse
import glob, json, os, re, shutil
from typing import Optional


def neural_run_group(neural_config: dict) -> Optional[str]:
    heuristic = neural_config["heuristic_context_selector"]

    heuristic_kwargs = neural_config["heuristic_context_selector_kwargs"]
    if "sents_nb" in heuristic_kwargs:
        heuristic_sents_nb = heuristic_kwargs["sents_nb"]
    elif "left_sents_nb" in heuristic_kwargs:
        left_sents_nb = heuristic_kwargs["left_sents_nb"]
        right_sents_nb = heuristic_kwargs["right_sents_nb"]
        if left_sents_nb == right_sents_nb:
            heuristic_sents_nb = left_sents_nb * 2
        else:
            if left_sents_nb > 0 and right_sents_nb == 0:
                heuristic = "left"
                heuristic_sents_nb = left_sents_nb
            elif right_sents_nb > 0 and left_sents_nb == 0:
                heuristic = "right"
                heuristic_sents_nb = right_sents_nb
            else:
                raise ValueError
    else:
        print(f"error processing {run_dir} heuristic kwargs. Ignoring...")
        return None

    return f"neural_{heuristic}_{heuristic_sents_nb}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--phase", type=str, default="phase1")
    parser.add_argument("-d", "--dry-run", action="store_true")
    args = parser.parse_args()

    run_dirs = [
        d
        for d in glob.glob(f"runs/{args.phase}/*")
        if re.match(r"[0-9]+", os.path.basename(d))
    ]

    for run_dir in run_dirs:

        with open(f"{run_dir}/config.json") as f:
            config = json.load(f)

        if "neural" in config["context_selectors"]:
            neural_config = config["context_selectors"]["neural"]
            run_group = neural_run_group(neural_config)
            if run_group is None:
                print(f"can't parse run {run_dir}. skipping...")
                continue
            sents_nb = neural_config["sents_nb"]
        elif "neighbors" in config["context_selectors"]:
            neighbors_config = config["context_selectors"]["neighbors"]
            left_sents_nb = neighbors_config["left_sents_nb"]
            right_sents_nb = neighbors_config["right_sents_nb"]
            if neighbors_config["left_sents_nb"] == 0:
                run_group = "right"
                sents_nb = right_sents_nb
            elif neighbors_config["right_sents_nb"] == 0:
                run_group = "left"
                sents_nb = left_sents_nb
            else:
                run_group = "neighbors"
                assert left_sents_nb == right_sents_nb
                sents_nb = left_sents_nb + right_sents_nb
        else:
            assert len(config["context_selectors"]) == 1
            run_group = list(config["context_selectors"].keys())[0]
            sents_nb = config["context_selectors"][run_group]["sents_nb"]

        run_group_dir = f"runs/{args.phase}/{run_group}"
        if not args.dry_run:
            os.makedirs(run_group_dir, exist_ok=True)

        print(f"moving {run_dir} to {run_group_dir}/{sents_nb}...")
        if not args.dry_run:
            shutil.move(run_dir, f"{run_group_dir}/{sents_nb}")
