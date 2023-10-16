import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scienceplots


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("-m", "--metrics", type=str, default="f1")
parser.add_argument(
    "-g",
    "--runs-group",
    type=str,
    default="unsupervised",
    help="Runs group to plot. Either 'unsupervised' or 'supervised'.",
)
args = parser.parse_args()

FONTSIZE = 10
COLUMN_WIDTH_IN = 3.0315

MARKERS = ["x", "+", "h", "*", "d", "p", "^"]

neural_run = {
    "name": "neural (our)",
    "metrics": f"mean_test_ner_{args.metrics}",
    "report_stdev": True,
}

if args.runs_group == "unsupervised":
    runs = {
        "book_neighbors": {
            "name": "surrounding",
            "metrics": f"mean_test_{args.metrics}",
        },
        "book_bm25": {"name": "bm25", "metrics": f"mean_test_{args.metrics}"},
        "book_samenoun": {
            "name": "samenoun",
            "metrics": f"mean_test_{args.metrics}",
            "report_stdev": True,
        },
        "neural_book_s7b_n8": neural_run,
    }
elif args.runs_group == "supervised":
    runs = {
        "monobert_bm25": {
            "name": "bm25+monobert",
            "metrics": f"mean_test_{args.metrics}",
            "report_stdev": True,
        },
        "monobert_all": {
            "name": "all+monobert",
            "metrics": f"mean_test_{args.metrics}",
            "report_stdev": True,
        },
        "monot5_bm25": {
            "name": "bm25+monot5",
            "metrics": f"mean_test_{args.metrics}",
            "report_stdev": True,
        },
        "monot5_all": {
            "name": "all+monot5",
            "metrics": f"mean_test_{args.metrics}",
            "report_stdev": True,
        },
        "neural_book_s7b_n8": neural_run,
    }
else:
    raise ValueError(
        f"unknown run group: {args.runs_group} (should be one of: 'unsupervised', 'supervised')"
    )

plt.style.use("science")
plt.rc("xtick", labelsize=FONTSIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONTSIZE)  # fontsize of the tick labels
fig, ax = plt.subplots()

fig.set_size_inches(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.6)

# plot baseline
with open("./runs/gen/gen_base_models/metrics.json") as f:
    bare_metrics = json.load(f)
ax.plot(
    [1, 8],
    [bare_metrics[f"mean_test_{args.metrics}"]["values"][0]] * 2,
    linewidth=1,
    c="black",
    label="no retrieval",
)

# plot runs in general
for run_i, (run, run_attrs) in enumerate(runs.items()):

    run_dir = f"./runs/gen/{run}"

    with open(f"{run_dir}/config.json") as f:
        config = json.load(f)

    with open(f"{run_dir}/metrics.json") as f:
        metrics = json.load(f)

    metrics_name = run_attrs["metrics"]

    if run_attrs.get("report_stdev"):
        run_metrics = []
        for run_i in range(config["runs_nb"]):
            run_metrics.append(metrics[f"run{run_i}.{metrics_name}"]["values"])
        stdev = np.std(np.array(run_metrics), axis=0)

    ax.errorbar(
        [int(step) for step in metrics[metrics_name]["steps"] if int(step) <= 8],
        metrics[metrics_name]["values"][:8],
        yerr=stdev[:8] if run_attrs.get("report_stdev") else None,
        capsize=3,
        marker=MARKERS[run_i],
        markersize=3,
        linewidth=1,
        label=run_attrs["name"],
    )

ax.legend(
    loc="lower center",
    ncol=len(runs) // 2,
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
