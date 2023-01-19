import argparse, json
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("-r", "--oracle", action="store_true")
args = parser.parse_args()


runs = ["random", "bm25", "samenoun", "left", "right", "neighbors"]

with open(f"./runs/bare/metrics.json") as f:
    bare_metrics = json.load(f)


plt.style.use("science")
# plt.rcParams.update({"xtick.labelsize": 18})
# plt.rcParams.update({"ytick.labelsize": 18})
plt.rc("xtick", labelsize=40)  # fontsize of the tick labels
plt.rc("ytick", labelsize=40)  # fontsize of the tick labels
fig, ax = plt.subplots()

fig.set_size_inches(16, 8)

for run in runs:
    if args.oracle:
        run = f"oracle_{run}"
    with open(f"./runs/short/{run}/metrics.json") as f:
        metrics = json.load(f)
    ax.plot(
        [int(step) for step in metrics["mean_test_f1"]["steps"]],
        metrics["mean_test_f1"]["values"],
    )

# bare baseline
ax.plot(
    [1, 6],
    [bare_metrics["mean_test_f1"]["values"][0]] * 2,
    linestyle="--",
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
