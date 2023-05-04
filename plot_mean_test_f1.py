import argparse, json
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("-r", "--oracle", action="store_true")
parser.add_argument("-e", "--restricted", action="store_true")
parser.add_argument("--no-baseline", action="store_true")
args = parser.parse_args()

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
}

with open(f"./runs/bare/metrics.json") as f:
    bare_metrics = json.load(f)


plt.style.use("science")
plt.rc("xtick", labelsize=40)  # fontsize of the tick labels
plt.rc("ytick", labelsize=40)  # fontsize of the tick labels
fig, ax = plt.subplots()

fig.set_size_inches(24, 16)

global_runs_artists = []
local_runs_artists = []
for run_i, (run, run_name) in enumerate(runs.items()):
    with open(f"./runs/short/{run}/metrics.json") as f:
        metrics = json.load(f)
    (p,) = ax.plot(
        [int(step) for step in metrics["mean_test_f1"]["steps"]],
        metrics["mean_test_f1"]["values"],
        marker=MARKERS[run_i],
        markersize=20,
        linewidth=4,
    )
    if runs_groups[run_name] == "global":
        global_runs_artists.append(p)
    else:
        local_runs_artists.append(p)

# no retrieval baseline
if not args.no_baseline:
    (no_retrieval_p,) = ax.plot(
        [1, 6],
        [bare_metrics["mean_test_f1"]["values"][0]] * 2,
        linewidth=4,
    )

ax.grid()
ax.set_ylabel("F1", fontsize=40)
ax.set_xlabel("Number of retrieved sentences", fontsize=40)

ncol = 2 if args.no_baseline else 3

legends = []

l1 = ax.legend(
    global_runs_artists,
    [r for r in runs.values() if runs_groups[r] == "global"],
    loc="lower center",
    bbox_to_anchor=(0, 1) if ncol == 3 else (0.1, 1),
    fontsize=40,
    title="global",
    title_fontsize=50,
    alignment="left",
    mode="expand",
)
legends.append(l1)

l2 = ax.legend(
    local_runs_artists,
    [r for r in runs.values() if runs_groups[r] == "local"],
    loc="lower center",
    bbox_to_anchor=(0.5, 1) if ncol == 3 else (0.7, 1),
    fontsize=40,
    title="local",
    title_fontsize=50,
    alignment="left",
)
legends.append(l2)

if not args.no_baseline:
    l3 = ax.legend(
        [no_retrieval_p],  # type: ignore
        ["no retrieval"],
        loc="lower center",
        bbox_to_anchor=(0.85, 1),
        fontsize=40,
        alignment="left",
    )
    legends.append(l3)

ax.add_artist(l1)
ax.add_artist(l2)


if args.output:
    plt.savefig(args.output, bbox_extra_artists=legends, bbox_inches="tight")
else:
    plt.show()
