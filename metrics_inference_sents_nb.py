from typing import Any, Dict, List, Tuple, Callable, Optional
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
    run_metrics: Dict[str, Dict[str, Any]], run_name: str
) -> Tuple[List[int], List[float], List[float], List[float]]:
    runs_nb = 5
    folds_nb = 5
    inference_sents_nb_list = list(range(8, 73, 8))
    sents_f1s = [[] for _ in inference_sents_nb_list]

    for run_i in range(runs_nb):
        for fold_i in range(folds_nb):
            for inference_sents_nb_i in range(len(inference_sents_nb_list)):
                sents_f1s[inference_sents_nb_i].append(
                    run_metrics[run_name][f"run{run_i}.fold{fold_i}.test_f1"]["values"][
                        inference_sents_nb_i
                    ]
                )

    return (
        inference_sents_nb_list,
        [mean(f1s) for f1s in sents_f1s],
        [stdev(f1s) for f1s in sents_f1s],
        [stdev(f1s) for f1s in sents_f1s],
    )


custom_metrics = [
    partial_with_name(test_f1_mean, "test_f1_mean", run_name="inference_sents_nb"),
    partial_with_name(
        test_f1_mean, "test_f1_mean_thg", run_name="inference_sents_nb_thg"
    ),
]
