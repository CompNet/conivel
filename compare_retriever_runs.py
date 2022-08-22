from __future__ import annotations
from typing import Dict, Literal, Tuple, List
import glob, json, argparse, math
from statistics import mean, stdev
import matplotlib.pyplot as plt


class RetrievalMethod:

    name: str

    def xtick_from_config(self, config: dict) -> int:
        raise NotImplementedError

    def display_name(self) -> str:
        return self.__class__.name


class RandomRetrievalMethod(RetrievalMethod):

    name = "random"

    def xtick_from_config(self, config: dict) -> int:
        return config["context_selectors"]["random"]["sents_nb"]


class LeftContextRetrievalMethod(RetrievalMethod):

    name = "left_context"

    def xtick_from_config(self, config: dict) -> int:
        return config["context_selectors"]["neighbors"]["left_sents_nb"]


class RightContextRetrievalMethod(RetrievalMethod):

    name = "right_context"

    def xtick_from_config(self, config: dict) -> int:
        return config["context_selectors"]["neighbors"]["right_sents_nb"]


class NeighborsContextRetrievalMethod(RetrievalMethod):

    name = "neighbors"

    def xtick_from_config(self, config: dict) -> int:
        return config["context_selectors"]["neighbors"]["left_sents_nb"] * 2

    def display_name(self) -> str:
        return "left + right"


class SameWordContextRetrievalMethod(RetrievalMethod):

    name = "sameword"

    def xtick_from_config(self, config: dict) -> int:
        return config["context_selectors"]["sameword"]["sents_nb"]

    def display_name(self) -> str:
        return "shared noun"


class NeuralContextRetrievalMethod(RetrievalMethod):

    name = "neural_context_random_6_heuristic_sents"

    def xtick_from_config(self, config: dict) -> int:
        return config["context_selectors"]["neural"]["sents_nb"]

    def display_name(self) -> str:
        return "neural (with 'random' heuristic)"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metric_name", type=str, default="dekker_f1")
    parser.add_argument(
        "-e",
        "--errorbars",
        type=str,
        default="stdev",
        help="one of : {'stdev', 'extremas'}",
    )
    args = parser.parse_args()

    # load base f1
    with open("./runs/no_context/metrics.json") as f:
        no_ctx_metrics = json.load(f)
    no_ctx_metrics = no_ctx_metrics[args.metric_name]["values"]

    # load f1s from context xps
    retrieval_methods = [
        RandomRetrievalMethod(),
        LeftContextRetrievalMethod(),
        RightContextRetrievalMethod(),
        NeighborsContextRetrievalMethod(),
        SameWordContextRetrievalMethod(),
        NeuralContextRetrievalMethod(),
    ]

    cols_nb = math.ceil(len(retrieval_methods) / 2)
    fig, axs = plt.subplots(2, cols_nb)

    for i, method in enumerate(retrieval_methods):

        datas: List[
            Dict[Literal["xtick", "mean", "stdev", "min", "max"], int | float]
        ] = []
        datas.append(
            {
                "xtick": 0,
                "mean": mean(no_ctx_metrics),
                "stdev": stdev(no_ctx_metrics),
                "min": min(no_ctx_metrics),
                "max": max(no_ctx_metrics),
            }
        )

        for run_dir in glob.glob(f"./runs/{method.name}/*"):

            print(f"processing {run_dir}...")

            with open(f"{run_dir}/config.json") as f:
                run_config = json.load(f)
            xtick = method.xtick_from_config(run_config)

            with open(f"{run_dir}/metrics.json") as f:
                run_metrics = json.load(f)

            metrics = run_metrics[args.metric_name]["values"]

            datas.append(
                {
                    "xtick": xtick,
                    "mean": mean(metrics),
                    "stdev": stdev(metrics),
                    "min": min(metrics),
                    "max": max(metrics),
                }
            )

        datas = sorted(datas, key=lambda d: d["xtick"])
        xticks = [d["xtick"] for d in datas]
        metrics_mean = [d["mean"] for d in datas]

        grid_i = i // cols_nb
        grid_j = i % cols_nb

        if args.errorbars == "stdev":
            metrics_stdev = [d["stdev"] for d in datas]
            axs[grid_i][grid_j].errorbar(
                xticks, metrics_mean, yerr=metrics_stdev, capsize=7
            )
        elif args.errorbars == "extremas":
            mins = [d["min"] for d in datas]
            mins_diff = [mean - min for mean, min in zip(metrics_mean, mins)]
            maxs = [d["max"] for d in datas]
            maxs_diff = [abs(max - mean) for mean, max in zip(metrics_mean, maxs)]
            axs[grid_i][grid_j].errorbar(
                xticks, metrics_mean, yerr=[mins_diff, maxs_diff], capsize=7
            )
        else:
            raise ValueError(f"no known errorbars method : {args.errorbars}")
        axs[grid_i][grid_j].set_title(method.display_name())
        axs[grid_i][grid_j].grid()

    plt.suptitle(args.metric_name)
    plt.show()
