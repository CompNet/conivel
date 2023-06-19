import argparse, json, os
import matplotlib.pyplot as plt
import scienceplots


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("-m", "--metrics", type=str, default="f1")
args = parser.parse_args()

FONTSIZE = 25

runs = {
    "full_bm25": {"name": "book", "metrics": f"mean_test_{args.metrics}"},
    "vanilla_bm25": {"name": "chapter", "metrics": f"mean_test_{args.metrics}"},
    "full_samenoun": {
        "name": "book",
        "metrics": f"mean_test_{args.metrics}",
    },
    "vanilla_samenoun": {
        "name": "chapter",
        "metrics": f"mean_test_{args.metrics}",
    },
    "full_gen_all_noT": {
        "name": "neural alpaca-7b",
        "metrics": f"mean_test_ner_{args.metrics}",
    },
}

for run_name, run_dict in runs.items():
    with open(f"./runs/gen/{run_name}/metrics.json") as f:
        metrics = json.load(f)
    run_dict["values"] = metrics[run_dict["metrics"]]["values"]


plt.style.use("science")
plt.rc("xtick", labelsize=FONTSIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONTSIZE)  # fontsize of the tick labels

fig, axs = plt.subplots(1, 3, figsize=(20, 5))


def plot_duo(ax, run_1: dict, run_2: dict, title: str):
    x = list(range(1, 7))
    ax.plot(x, run_1["values"], label=run_1["name"], linewidth=3, marker="o")
    ax.plot(x, run_2["values"], label=run_2["name"], linewidth=3, marker="v")
    ax.legend(fontsize=FONTSIZE)
    ax.grid()
    ax.set_xlabel("Max number of retrieved sentences", fontsize=FONTSIZE)
    ax.set_ylabel("F1", fontsize=FONTSIZE)
    ax.set_title(title, fontsize=FONTSIZE)


plot_duo(axs[0], runs["full_bm25"], runs["vanilla_bm25"], "bm25")
plot_duo(axs[1], runs["full_samenoun"], runs["vanilla_samenoun"], "samenoun")

plt.tight_layout()

if args.output:
    plt.savefig(os.path.expanduser(args.output), bbox_inches="tight")
else:
    plt.show()
