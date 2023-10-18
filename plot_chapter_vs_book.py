from typing import List
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

runs = {
    "book_bm25": {"name": "book", "metrics": f"mean_test_{args.metrics}"},
    "chapter_bm25": {"name": "chapter", "metrics": f"mean_test_{args.metrics}"},
    "book_samenoun": {
        "name": "book",
        "metrics": f"mean_test_{args.metrics}",
        "report_stdev": True,
    },
    "chapter_samenoun": {
        "name": "chapter",
        "metrics": f"mean_test_{args.metrics}",
        "report_stdev": True,
    },
    "neural_book_s7b_n8": {
        "name": "book",
        "metrics": f"mean_test_ner_{args.metrics}",
        "report_stdev": True,
    },
    "neural_chapter_s7b_n8": {
        "name": "chapter",
        "metrics": f"mean_test_ner_{args.metrics}",
        "report_stdev": True,
    },
    "neural_book_s7b_n4": {
        "name": "book",
        "metrics": f"mean_test_ner_{args.metrics}",
        "report_stdev": True,
    },
    "neural_chapter_s7b_n4": {
        "name": "chapter",
        "metrics": f"mean_test_ner_{args.metrics}",
        "report_stdev": True,
    },
    "neural_book_s7b_n12": {
        "name": "book",
        "metrics": f"mean_test_ner_{args.metrics}",
        "report_stdev": True,
    },
    "neural_chapter_s7b_n12": {
        "name": "chapter",
        "metrics": f"mean_test_ner_{args.metrics}",
        "report_stdev": True,
    },
}

for run_name, run_dict in runs.items():

    run_dir = f"./runs/gen/{run_name}"

    with open(f"{run_dir}/config.json") as f:
        config = json.load(f)

    with open(f"{run_dir}/metrics.json") as f:
        metrics = json.load(f)

    metrics_name = run_dict["metrics"]

    try:
        run_dict["values"] = metrics[metrics_name]["values"]

        if run_dict.get("report_stdev"):
            run_metrics = []
            for run_i in range(config["runs_nb"]):
                run_metrics.append(metrics[f"run{run_i}.{metrics_name}"]["values"])
            run_dict["stdev"] = np.std(np.array(run_metrics), axis=0)

    except KeyError:
        print(f"error reading run {run_name}")
        exit(1)


plt.style.use("science")
plt.rc("xtick", labelsize=FONTSIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONTSIZE)  # fontsize of the tick labels

fig, axs = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * ASPECT_RATIO))


def plot_duo(ax, run_1: dict, run_2: dict, title: str):
    x = list(range(1, 9))
    run_1_stdev = run_1.get("stdev")
    ax.errorbar(
        x,
        run_1["values"][:8],
        yerr=None if run_1_stdev is None else run_1_stdev[:8],
        label=run_1["name"],
        capsize=3,
        linewidth=1,
        marker="x",
        markersize=4,
    )
    run_2_stdev = run_2.get("stdev")
    ax.errorbar(
        x,
        run_2["values"][:8],
        yerr=None if run_2_stdev is None else run_2_stdev[:8],
        label=run_2["name"],
        capsize=3,
        linewidth=1,
        marker="+",
        markersize=4,
    )
    ax.grid()
    ax.set_ylabel("F1", fontsize=FONTSIZE)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_multiples(ax, book_runs: List[dict], chapter_runs: List[dict], title: str):

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for run in book_runs:
        run_stdev = run.get("stdev")
        ax.errorbar(
            list(range(1, 9)),
            run["values"][:8],
            yerr=None if run_stdev is None else run_stdev[:8],
            # HACK: no label here. We hope that the other plots
            # already set up labels...
            # label=run["name"],
            capsize=3,
            linewidth=1,
            marker="x",
            markersize=4,
            c=colors[0],
        )

    for run in chapter_runs:
        run_stdev = run.get("stdev")
        ax.errorbar(
            list(range(1, 9)),
            run["values"][:8],
            yerr=None if run_stdev is None else run_stdev[:8],
            # HACK: no label here. We hope that the other plots
            # already set up labels...
            # label=run["name"],
            capsize=3,
            linewidth=1,
            marker="+",
            markersize=4,
            c=colors[1],
        )

    ax.grid()
    ax.set_ylabel("F1", fontsize=FONTSIZE)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


plot_duo(axs[0], runs["book_bm25"], runs["chapter_bm25"], "bm25")
plot_duo(axs[1], runs["book_samenoun"], runs["chapter_samenoun"], "samenoun")
plot_multiples(
    axs[2],
    [
        runs["neural_book_s7b_n4"],
        runs["neural_book_s7b_n8"],
        runs["neural_book_s7b_n12"],
    ],
    [
        runs["neural_chapter_s7b_n4"],
        runs["neural_chapter_s7b_n8"],
        runs["neural_chapter_s7b_n12"],
    ],
    "neural (our)",
)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.4, 1.1), fontsize=FONTSIZE, ncol=2)
fig.text(
    0.5, -0.05, "Number of retrieved sentences $k$", ha="center", fontsize=FONTSIZE
)
plt.tight_layout()

if args.output:
    plt.savefig(os.path.expanduser(args.output), bbox_inches="tight")
else:
    plt.show()
