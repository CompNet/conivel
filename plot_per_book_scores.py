import json, argparse, os
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-m", "--metrics", type=str, default="f1")
    args = parser.parse_args()

    FONTSIZE = 15

    with open("./runs/gen/gen_base_models/info.json") as f:
        doc_names = json.load(f)["documents_names"]
    pretty_doc_names = [os.path.splitext(name)[0] for name in doc_names]

    runs: Dict[str, Dict[str, Any]] = {
        "gen_base_models": {"name": "no retrieval", "color": "black"},
        "vanilla_neighbors": {"name": "surrounding", "color": "tab:blue"},
        "full_bm25": {"name": "bm25", "color": "tab:green"},
        "full_samenoun": {"name": "samenoun", "color": "tab:orange"},
        "full_gen_all_noT": {"name": "neural alpaca-7b", "color": "tab:red"},
    }

    keys = [f"mean_{doc_name}_test_{args.metrics}" for doc_name in doc_names]
    for run, run_dict in runs.items():
        with open(f"./runs/gen/{run}/metrics.json") as f:
            metrics = json.load(f)
        run_dict["values"] = [metrics[key]["values"][0] for key in keys]

    # number of time each method is better or equal to its
    # counterparts
    maxs_count = [0] * len(runs)
    for values in zip(*[run_dict["values"] for run_dict in runs.values()]):
        values = np.array(values)
        maxs_i = np.where(values == max(values))[0]
        for i in maxs_i:
            maxs_count[i] += 1
    for name, max_value in zip(
        [run_dict["name"] for run_dict in runs.values()], maxs_count
    ):
        print(f"{name}: {max_value}")

    # max enhancement of the neural method compared to no retrieval
    max_enhancement = max(
        [
            neural - bare
            for bare, neural in zip(
                runs["gen_base_models"]["values"], runs["full_gen_all_noT"]["values"]
            )
        ]
    )
    print(f"neural method max enhancement : {max_enhancement}")

    # per-book F1 plot
    plt.style.use("science")

    fig, ax = plt.subplots(figsize=(10, 4.5))

    width = 1 / (2 * len(runs))
    x = np.array(list(range(len(doc_names))))
    for i, run_dict in enumerate(runs.values()):
        offset = width * i
        rects = ax.bar(
            x + offset,
            run_dict["values"],
            width,
            color=run_dict["color"],
            label=run_dict["name"],
        )

    ax.set_xticks(
        x + width, [os.path.splitext(name)[0] for name in doc_names], rotation=90
    )
    ax.set_ylabel("F1 score", fontsize=FONTSIZE)
    ax.set_ylim(0.65, 1.0)
    ax.legend(
        loc="lower center",
        ncol=len(runs) // 2,
        bbox_to_anchor=(0.5, 1),
        fontsize=FONTSIZE,
    )

    plt.tight_layout()

    if args.output:
        plt.savefig(os.path.expanduser(args.output))
    else:
        plt.show()
