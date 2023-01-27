import argparse, json
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument(
    "-g", "--group", type=str, default="global", help="one of: 'global', 'local'"
)
args = parser.parse_args()


if args.group == "global":
    runs = [
        ["random", "neural_random", "neural_ideal_random"],
        ["bm25", "neural_bm25", "neural_ideal_bm25"],
        ["samenoun", "neural_samenoun", "neural_ideal_samenoun"],
    ]
elif args.group == "local":
    runs = [
        ["left", "neural_left", "neural_ideal_left"],
        ["right", "neural_right", "neural_ideal_right"],
        ["neighbors", "neural_neighbors", "neural_ideal_neighbors"],
    ]
else:
    raise ValueError(f"unknown group: {args.group}")

with open(f"./runs/bare/metrics.json") as f:
    bare_metrics = json.load(f)


plt.style.use("science")
plt.rcParams.update({"xtick.labelsize": 18})
plt.rcParams.update({"ytick.labelsize": 18})
fig, axs = plt.subplots(1, 3)

fig.set_size_inches(24, 4)

for i, run_group in enumerate(runs):

    min_steps = []
    max_steps = []
    for run in run_group:
        with open(f"./runs/{run}/metrics.json") as f:
            metrics = json.load(f)
        axs[i].plot(
            [int(step) for step in metrics["mean_test_f1"]["steps"]],
            metrics["mean_test_f1"]["values"],
        )
        min_steps.append(min(metrics["mean_test_f1"]["steps"]))
        max_steps.append(max(metrics["mean_test_f1"]["steps"]))

    # bare baseline
    axs[i].plot(
        [min(min_steps), max(max_steps)],
        [bare_metrics["mean_test_f1"]["values"][0]] * 2,
        linestyle="--",
    )

    axs[i].grid()
    axs[i].set_ylabel("F1", fontsize=20)
    axs[i].set_xlabel("Number of retrieved sentences", fontsize=20)
    axs[i].legend(
        [r.replace("_", " ") for r in run_group] + ["no retrieval"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        fontsize=20,
    )


if args.output:
    plt.savefig(args.output)
else:
    plt.show()
