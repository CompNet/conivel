from typing import Any, Dict, List, Tuple, Callable, Optional, runtime_checkable
import functools
from statistics import mean, stdev


def partial_with_name(
    fn: Callable, name: Optional[str] = None, *args, **kwargs
) -> Callable:
    partial_fn = functools.partial(fn, *args, **kwargs)
    if name is None:
        name = fn.__name__
    setattr(partial_fn, "__name__", name)
    return partial_fn


def test_f1_mean(
    run_metrics: Dict[str, Dict[str, Any]],
    run_name: str,
    sents_nb_list: Optional[List] = None,
    is_ganged: bool = False,
    runs_nb: int = 5,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    if sents_nb_list is None:
        sents_nb_list = list(range(1, 9))
    folds_nb = 5

    sents_f1s = [[] for _ in sents_nb_list]
    for run_i in range(runs_nb):
        for fold_i in range(folds_nb):
            for sent_k in range(len(sents_nb_list)):
                if is_ganged:
                    sents_f1s[sent_k].append(
                        run_metrics[run_name][
                            f"run{run_i}.fold{fold_i}.test_f1.fold{fold_i}"
                        ]["values"][sent_k]
                    )
                else:
                    sents_f1s[sent_k].append(
                        run_metrics[run_name][f"run{run_i}.fold{fold_i}.test_f1"][
                            "values"
                        ][sent_k]
                    )

    return (
        sents_nb_list,
        [mean(f1s) for f1s in sents_f1s],
        [stdev(f1s) for f1s in sents_f1s],
        [stdev(f1s) for f1s in sents_f1s],
    )


custom_metrics = [
    partial_with_name(test_f1_mean, "random_test_f1_mean", run_name="random"),
    partial_with_name(test_f1_mean, "bm25_test_f1_mean", run_name="bm25"),
    partial_with_name(test_f1_mean, "left_test_f1_mean", run_name="left"),
    partial_with_name(test_f1_mean, "right_test_f1_mean", run_name="right"),
    partial_with_name(
        test_f1_mean,
        "neighbors_test_f1_mean",
        run_name="neighbors",
        sents_nb_list=[2, 4, 6, 8],
    ),
    partial_with_name(test_f1_mean, "samenoun_test_f1_mean", run_name="samenoun"),
    partial_with_name(
        test_f1_mean,
        "neural_random_test_f1_mean",
        run_name="neural_random",
        is_ganged=True,
    ),
    partial_with_name(
        test_f1_mean, "neural_bm25_test_f1_mean", run_name="neural_bm25", is_ganged=True
    ),
    partial_with_name(
        test_f1_mean,
        "neural_bm25_restricted_test_f1_mean",
        run_name="neural_bm25_restricted",
        sents_nb_list=list(range(1, 7)),
    ),
    partial_with_name(
        test_f1_mean,
        "neural_bm25_2_test_f1_mean",
        run_name="neural_bm25_2",
        sents_nb_list=list(range(1, 7)),
        runs_nb=2,
    ),
    partial_with_name(
        test_f1_mean,
        "neural_samenoun_test_f1_mean",
        run_name="neural_samenoun",
        is_ganged=True,
    ),
    partial_with_name(
        test_f1_mean,
        "neural_samenoun_highsentsnb_test_f1_mean",
        run_name="neural_samenoun_highsentsnb",
        is_ganged=True,
    ),
    partial_with_name(
        test_f1_mean,
        "neural_samenoun_highsentsnb2_test_f1_mean",
        run_name="neural_samenoun_highsentsnb2",
        is_ganged=True,
    ),
    partial_with_name(
        test_f1_mean,
        "neural_samenoun_highsentsnb2_thehungergames_test_f1_mean",
        run_name="neural_samenoun_highsentsnb2_thehungergames",
    ),
]
