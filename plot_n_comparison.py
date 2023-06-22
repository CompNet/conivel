import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scienceplots


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("-m", "--metrics", type=str, default="f1")
args = parser.parse_args()

FONTSIZE = 10
TEXT_WIDTH_IN = 6.29921
ASPECT_RATIO = 0.25
MARKERS = ["x", "+", "h", "*", "d", "p", "^"]

models = ["7b", "13b"]
n_list = [4, 8, 12, 16, 24]


plot_data = {n: {model: {} for model in models} for n in n_list}


for model in models:

    for n in n_list:

        run_dir = f"./runs/gen/neural_book_s{model}_n{n}"

        with open(f"{run_dir}/config.json") as f:
            config = json.load(f)

        with open(f"{run_dir}/metrics.json") as f:
            metrics = json.load(f)

        metrics_key = f"mean_test_ner_{args.metrics}"

        try:
            plot_data[n][model]["values"] = metrics[metrics_key]["values"]
            plot_data[n][model]["steps"] = metrics[metrics_key]["steps"]

            run_metrics = []
            for run_i in range(config["runs_nb"]):
                run_metrics.append(metrics[f"run{run_i}.{metrics_key}"]["values"])
            plot_data[n][model]["stdev"] = np.std(np.array(run_metrics), axis=0)

        except KeyError as e:
            print(f"error reading run {run_dir}: {e}")
            exit(1)


plt.style.use("science")
plt.rc("xtick", labelsize=FONTSIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONTSIZE)  # fontsize of the tick labels

fig, axs = plt.subplots(
    1,
    len(plot_data),
    figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * ASPECT_RATIO),
    sharex=True,
    sharey=True,
)
axs[0].set_ylabel("F1", fontsize=FONTSIZE)

for i, (n, model_data) in enumerate(plot_data.items()):
    for model_i, model in enumerate(models):
        axs[i].errorbar(
            model_data[model]["steps"],
            model_data[model]["values"],
            yerr=model_data[model]["stdev"],
            label=f"alpaca-{model}",
            capsize=3,
            linewidth=1,
            marker=MARKERS[model_i],
            markersize=4,
        )
        axs[i].set_title(f"n = {n}", fontsize=FONTSIZE)
    axs[i].grid()
    axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))

handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.4, 1.2), fontsize=FONTSIZE, ncol=2)
fig.text(
    0.5, -0.05, "Number of retrieved sentences $k$", ha="center", fontsize=FONTSIZE
)
plt.tight_layout()

if args.output:
    plt.savefig(os.path.expanduser(args.output), bbox_inches="tight")
else:
    plt.show()
