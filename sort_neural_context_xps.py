import glob, json, os, re, shutil


if __name__ == "__main__":

    run_dirs = [
        d for d in glob.glob("runs/*") if re.match(r"[0-9]+", os.path.basename(d))
    ]

    for run_dir in run_dirs:

        with open(f"{run_dir}/config.json") as f:
            config = json.load(f)

        neural_config = config["context_selectors"]["neural"]

        heuristic = neural_config["heuristic_context_selector"]

        heuristic_kwargs = neural_config["heuristic_context_selector_kwargs"]
        if "sents_nb" in heuristic_kwargs:
            heuristic_sents_nb = heuristic_kwargs["sents_nb"]
        elif "left_sents_nb" in heuristic_kwargs:
            assert (
                heuristic_kwargs["left_sents_nb"] == heuristic_kwargs["right_sents_nb"]
            )
            heuristic_sents_nb = heuristic_kwargs["left_sents_nb"] * 2
        else:
            print(f"error processing {run_dir} heuristic kwargs. Ignoring...")
            continue

        sents_nb = neural_config["sents_nb"]

        run_group_dir = f"runs/neural_context_{heuristic}_{heuristic_sents_nb}"
        os.makedirs(run_group_dir, exist_ok=True)

        print(f"moving {run_dir} to {run_group_dir}/{sents_nb}...")
        shutil.move(run_dir, f"{run_group_dir}/{sents_nb}")
