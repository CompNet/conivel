import argparse, json, os
import matplotlib.pyplot as plt
import scienceplots


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("-m", "--metrics", type=str, default="f1")
args = parser.parse_args()

FONTSIZE = 25

runs = {
    "book_bm25": {"name": "book", "metrics": f"mean_test_{args.metrics}"},
    "chapter_bm25": {"name": "chapter", "metrics": f"mean_test_{args.metrics}"},
    "book_samenoun": {
        "name": "book",
        "metrics": f"mean_test_{args.metrics}",
    },
    "chapter_samenoun": {
        "name": "chapter",
        "metrics": f"mean_test_{args.metrics}",
    },
    "neural_book_s13b_n8": {
        "name": "book",
        "metrics": f"mean_test_ner_{args.metrics}",
    },
    "neural_chapter_s13b_n8": {
        "name": "chapter",
        "metrics": f"mean_test_ner_{args.metrics}",
    },
}

for run_name, run_dict in runs.items():
    with open(f"./runs/gen/{run_name}/metrics.json") as f:
        metrics = json.load(f)
    try:
        run_dict["values"] = metrics[run_dict["metrics"]]["values"]
    except KeyError:
        print(f"error reading run {run_name}")
        # default debug value
        run_dict["values"] = [1.0] * 6  # type: ignore


plt.style.use("science")
plt.rc("xtick", labelsize=FONTSIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONTSIZE)  # fontsize of the tick labels

fig, axs = plt.subplots(1, 3, figsize=(20, 5))


def plot_duo(ax, run_1: dict, run_2: dict, title: str):
    x = list(range(1, 7))
    ax.plot(x, run_1["values"], label=run_1["name"], linewidth=3, marker="o")
    ax.plot(x, run_2["values"], label=run_2["name"], linewidth=3, marker="v")
    # ax.legend(fontsize=FONTSIZE)
    ax.grid()
    ax.set_ylabel("F1", fontsize=FONTSIZE)
    ax.set_title(title, fontsize=FONTSIZE)


plot_duo(axs[0], runs["book_bm25"], runs["chapter_bm25"], "bm25")
plot_duo(axs[1], runs["book_samenoun"], runs["chapter_samenoun"], "samenoun")
plot_duo(axs[2], runs["neural_book_s13b_n8"], runs["neural_chapter_s13b_n8"], "neural")

handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.3, 1.1), fontsize=FONTSIZE, ncol=2)
fig.text(
    0.5, -0.05, "Max number of retrieved sentences", ha="center", fontsize=FONTSIZE
)
plt.tight_layout()

if args.output:
    plt.savefig(os.path.expanduser(args.output), bbox_inches="tight")
else:
    plt.show()
