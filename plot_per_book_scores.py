import json, argparse, os
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

    with open("./runs/gen/gen_base_models/metrics.json") as f:
        bare_metrics = json.load(f)

    with open("./runs/gen/vanilla_neighbors/metrics.json") as f:
        surrounding_metrics = json.load(f)

    with open("./runs/gen/full_gen_all/metrics.json") as f:
        full_neural_all_metrics = json.load(f)

    with open("./runs/gen/full_bm25/metrics.json") as f:
        full_bm25_metrics = json.load(f)

    with open("./runs/gen/full_samenoun/metrics.json") as f:
        full_samenoun_metrics = json.load(f)

    keys = [f"mean_{doc_name}_test_{args.metrics}" for doc_name in doc_names]
    bare_y = [bare_metrics[key]["values"][0] for key in keys]
    surrounding_y = [surrounding_metrics[key]["values"][0] for key in keys]
    full_bm25_y = [full_bm25_metrics[key]["values"][0] for key in keys]
    full_samenoun_y = [full_samenoun_metrics[key]["values"][0] for key in keys]
    full_neural_all_y = [full_neural_all_metrics[key]["values"][0] for key in keys]

    # number of time each method is better or equal to its
    # counterparts
    maxs_count = [0] * 5
    for values in zip(
        bare_y, surrounding_y, full_bm25_y, full_samenoun_y, full_neural_all_y
    ):
        values = np.array(values)
        maxs_i = np.where(values == max(values))[0]
        for i in maxs_i:
            maxs_count[i] += 1
    for name, max_value in zip(
        ["no retrieval", "surrounding", "bm25", "samenoun", "neural"], maxs_count
    ):
        print(f"{name}: {max_value}")

    # max enhancement of the neural method compared to no retrieval
    max_enhancement = max(
        [neural - bare for bare, neural in zip(bare_y, full_neural_all_y)]
    )
    print(f"neural method max enhancement : {max_enhancement}")

    # per-book F1 plot
    plt.style.use("science")

    plot_data = {
        "no retrieval": {"values": bare_y, "color": "black"},
        "surrounding": {"values": surrounding_y, "color": "tab:blue"},
        "bm25": {"values": full_bm25_y, "color": "tab:green"},
        "samenoun": {"values": full_samenoun_y, "color": "tab:orange"},
        "neural": {"values": full_neural_all_y, "color": "tab:red"},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 1 / (2 * len(plot_data))
    x = np.array(list(range(len(doc_names))))
    for i, (label, data) in enumerate(plot_data.items()):
        offset = width * i
        rects = ax.bar(
            x + offset, data["values"], width, color=data["color"], label=label
        )

    ax.set_xticks(
        x + width, [os.path.splitext(name)[0] for name in doc_names], rotation=90
    )
    ax.set_ylabel("F1 score", fontsize=FONTSIZE)
    ax.legend(
        loc="lower center",
        ncols=len(plot_data),
        bbox_to_anchor=(0.5, 1),
        fontsize=FONTSIZE,
    )

    plt.tight_layout()

    if args.output:
        plt.savefig(os.path.expanduser(args.output))
    else:
        plt.show()
