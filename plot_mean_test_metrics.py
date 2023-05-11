import argparse, json, os
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("-r", "--oracle", action="store_true")
parser.add_argument("-e", "--restricted", action="store_true")
parser.add_argument("-m", "--metrics", type=str, default="f1")
parser.add_argument("--no-baseline", action="store_true")
args = parser.parse_args()

FONTSIZE = 25
MARKERS = ["o", "v", "^", "p", "s", "*", "D"]

# runs is of form {run_dir_name => name}
if args.oracle:
    runs = {
        f"oracle_{run}": run
        for run in ["random", "bm25", "samenoun", "before", "after", "surrounding"]
    }
elif args.restricted:
    runs = {"oracle_bm25": "bm25", "bm25_restricted": "restricted bm25"}
else:
    runs = {
        run: run
        for run in ["random", "bm25", "samenoun", "before", "after", "surrounding"]
    }
runs_groups = {
    "random": "global",
    "bm25": "global",
    "samenoun": "global",
    "before": "local",
    "after": "local",
    "surrounding": "local",
    "restricted bm25": "global",
}

with open(f"./runs/bare/metrics.json") as f:
    bare_metrics = json.load(f)


plt.style.use("science")
plt.rc("xtick", labelsize=FONTSIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONTSIZE)  # fontsize of the tick labels
fig, ax = plt.subplots()

fig.set_size_inches(9, 6)

global_runs_artists = []
local_runs_artists = []
for run_i, (run, run_name) in enumerate(runs.items()):
    with open(f"./runs/short/{run}/metrics.json") as f:
        metrics = json.load(f)
    (p,) = ax.plot(
        [int(step) for step in metrics[f"mean_test_{args.metrics}"]["steps"]],
        metrics[f"mean_test_{args.metrics}"]["values"],
        marker=MARKERS[run_i],
        markersize=8,
        linewidth=3,
    )
    if runs_groups[run_name] == "global":
        global_runs_artists.append(p)
    else:
        local_runs_artists.append(p)

# no retrieval baseline
if not args.no_baseline:
    (no_retrieval_p,) = ax.plot(
        [1, 6],
        [bare_metrics[f"mean_test_{args.metrics}"]["values"][0]] * 2,
        linewidth=3,
        c="black",
    )

ax.grid()
ax.set_ylabel(args.metrics.capitalize(), fontsize=FONTSIZE)
ax.set_xlabel("Max number of retrieved sentences", fontsize=FONTSIZE)

ncol = 2 if args.no_baseline or args.restricted else 3

legends = []

l1 = ax.legend(
    global_runs_artists,
    [r for r in runs.values() if runs_groups[r] == "global"],
    loc="lower center",
    bbox_to_anchor=(-0.1, 1) if ncol == 3 else (0, 1),
    fontsize=FONTSIZE,
    mode="expand",
)
legends.append(l1)

l2 = ax.legend(
    local_runs_artists,
    [r for r in runs.values() if runs_groups[r] == "local"],
    loc="lower center",
    bbox_to_anchor=(0.5, 1) if ncol == 3 else (0.7, 1),
    fontsize=FONTSIZE,
)
legends.append(l2)

if not args.no_baseline:
    l3 = ax.legend(
        [no_retrieval_p],  # type: ignore
        ["no retrieval"],
        loc="lower center",
        bbox_to_anchor=(0.95, 1) if ncol == 3 else (0.8, 1),
        fontsize=FONTSIZE,
    )
    legends.append(l3)

ax.add_artist(l1)
ax.add_artist(l2)


if args.output:
    plt.savefig(
        os.path.expanduser(args.output), bbox_extra_artists=legends, bbox_inches="tight"
    )
else:
    plt.show()
