import argparse, json, os
import matplotlib.pyplot as plt
import scienceplots


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("-m", "--metrics", type=str, default="f1")
args = parser.parse_args()

FONTSIZE = 25

models = ["7b", "13b"]
n_list = [4, 8, 12, 16, 24]


plot_data = {n: {model: {} for model in models} for n in n_list}


for model in models:

    for n in n_list:

        with open(f"./runs/gen/neural_book_s{model}_n{n}/metrics.json") as f:
            metrics = json.load(f)

        metrics_key = f"mean_test_ner_{args.metrics}"
        plot_data[n][model]["values"] = metrics[metrics_key]["values"]
        plot_data[n][model]["steps"] = metrics[metrics_key]["steps"]


plt.style.use("science")
plt.rc("xtick", labelsize=FONTSIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONTSIZE)  # fontsize of the tick labels

fig, axs = plt.subplots(1, len(plot_data), figsize=(20, 3), sharex=True, sharey=True)
axs[0].set_ylabel("F1", fontsize=FONTSIZE)

for i, (n, model_data) in enumerate(plot_data.items()):
    for model in models:
        axs[i].plot(
            model_data[model]["steps"],
            model_data[model]["values"],
            label=f"alpaca-{model}",
            linewidth=3,
        )
        axs[i].set_title(f"n = {n}", fontsize=FONTSIZE)
    axs[i].grid()

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
