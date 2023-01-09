import argparse, json
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
args = parser.parse_args()


runs = [
    ["random", "neural_random", "neural_ideal_random"],
    ["bm25", "neural_bm25", "neural_ideal_bm25"],
    ["samenoun", "neural_samenoun", "neural_ideal_samenoun"],
]

fig, axs = plt.subplots(1, 3)
for i, run_group in enumerate(runs):
    for run in run_group:
        with open(f"./runs/{run}/metrics.json") as f:
            metrics = json.load(f)
        axs[i].plot(metrics["mean_test_f1"]["steps"], metrics["mean_test_f1"]["values"])
        axs[i].grid()
        axs[i].set_ylabel("F1")
        axs[i].set_xlabel("Number of retrieved sentences")
        axs[i].legend(
            [r.replace("_", " ") for r in run_group],
            loc="lower center",
            bbox_to_anchor=(0.5, 1),
        )

plt.style.use("science")
if args.output:
    plt.savefig(args.output)
else:
    plt.show()
