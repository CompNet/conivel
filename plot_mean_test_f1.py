import argparse, json
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("-r", "--oracle", action="store_true")
args = parser.parse_args()

# from https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
linestyle_tuple = [
    ("solid", "solid"),
    ("dashed", "dashed"),
    ("dashdot", "dashdot"),
    ("densely dotted", (0, (1, 1))),
    ("long dash with offset", (5, (10, 3))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashed", (0, (5, 10))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("dotted", "dotted"),
    ("loosely dotted", (0, (1, 10))),
]


runs = ["random", "bm25", "samenoun", "left", "right", "neighbors"]

with open(f"./runs/bare/metrics.json") as f:
    bare_metrics = json.load(f)


plt.style.use("science")
plt.rc("xtick", labelsize=40)  # fontsize of the tick labels
plt.rc("ytick", labelsize=40)  # fontsize of the tick labels
fig, ax = plt.subplots()

fig.set_size_inches(16, 12)

for run_i, run in enumerate(runs):
    if args.oracle:
        run = f"oracle_{run}"
    with open(f"./runs/short/{run}/metrics.json") as f:
        metrics = json.load(f)
    ax.plot(
        [int(step) for step in metrics["mean_test_f1"]["steps"]],
        metrics["mean_test_f1"]["values"],
        linestyle=linestyle_tuple[run_i][1],
        linewidth=4,
    )

# bare baseline
ax.plot(
    [1, 6],
    [bare_metrics["mean_test_f1"]["values"][0]] * 2,
    linestyle=linestyle_tuple[len(runs)][1],
    linewidth=4,
)

ax.grid()
ax.set_ylabel("F1", fontsize=40)
ax.set_xlabel("Number of retrieved sentences", fontsize=40)
ax.legend(
    runs + ["no retrieval"],
    loc="lower center",
    bbox_to_anchor=(0.5, 1),
    fontsize=40,
    ncol=3,
)


if args.output:
    plt.savefig(args.output)
else:
    plt.show()
