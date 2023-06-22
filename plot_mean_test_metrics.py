import argparse, json, os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scienceplots


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("-m", "--metrics", type=str, default="f1")
args = parser.parse_args()

FONTSIZE = 10
COLUMN_WIDTH_IN = 3.0315

MARKERS = ["x", "+", "h", "*", "d", "p", "^"]

runs = {
    "chapter_neighbors": {
        "name": "surrounding",
        "metrics": f"mean_test_{args.metrics}",
    },
    "book_bm25": {"name": "bm25", "metrics": f"mean_test_{args.metrics}"},
    "book_samenoun": {"name": "samenoun", "metrics": f"mean_test_{args.metrics}"},
    "neural_book_s13b_n8": {
        "name": "neural",
        "metrics": f"mean_test_ner_{args.metrics}",
    },
}

plt.style.use("science")
plt.rc("xtick", labelsize=FONTSIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONTSIZE)  # fontsize of the tick labels
fig, ax = plt.subplots()

fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)

# plot baseline
with open("./runs/gen/gen_base_models/metrics.json") as f:
    bare_metrics = json.load(f)
ax.plot(
    [1, 6],
    [bare_metrics[f"mean_test_{args.metrics}"]["values"][0]] * 2,
    linewidth=1,
    c="black",
    label="no retrieval",
)

# plot runs in general
for run_i, (run, run_attrs) in enumerate(runs.items()):
    with open(f"./runs/gen/{run}/metrics.json") as f:
        metrics = json.load(f)
    ax.plot(
        [int(step) for step in metrics[run_attrs["metrics"]]["steps"]],
        metrics[run_attrs["metrics"]]["values"],
        marker=MARKERS[run_i],
        markersize=3,
        linewidth=1,
        label=run_attrs["name"],
    )

ax.legend(
    loc="lower center",
    ncol=(len(runs) + 1) // 2,
    bbox_to_anchor=(0.5, 1),
    fontsize=FONTSIZE,
)
ax.grid()
ax.set_ylabel(args.metrics.capitalize(), fontsize=FONTSIZE)
ax.set_xlabel("Number of retrieved sentences $k$", fontsize=FONTSIZE)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))


if args.output:
    plt.savefig(os.path.expanduser(args.output), bbox_inches="tight")
else:
    plt.show()
