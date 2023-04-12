import json
import os
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from conivel.datas.context import (
    ContextRetrievalDataset,
    ContextRetrievalExample,
    NeuralContextRetriever,
)
from conivel.utils import (
    RunLogScope,
    sacred_archive_huggingface_model,
    gpu_memory_usage,
    sacred_log_series,
)


script_dir = os.path.abspath(os.path.dirname(__file__))

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))
if os.path.isfile(f"{script_dir}/telegram_observer_config.json"):
    ex.observers.append(
        TelegramObserver.from_config(f"{script_dir}/telegram_observer_config.json")
    )


def cr_dataset_from_path(path: str) -> ContextRetrievalDataset:
    with open(path) as f:
        data = json.load(f)
    return ContextRetrievalDataset([ContextRetrievalExample(**ex) for ex in data])


@ex.config
def config():
    # -- common parameters
    batch_size: int
    # wether models should be saved or not
    save_models: bool = True
    # number of experiment repeats
    runs_nb: int

    # -- context retrieval parameters
    # number of epochs for context retrieval training
    ctx_retrieval_epochs_nb: int = 3
    # learning rate for context retrieval training
    ctx_retrieval_lr: float = 2e-5
    # dropout for context retriever
    ctx_retrieval_dropout: float = 0.1

    # -- context retrieval dataset
    cr_test_dataset_path: str
    cr_train_dataset_path: str


@ex.automain
def main(
    _run: Run,
    batch_size: int,
    save_models: bool,
    runs_nb: int,
    ctx_retrieval_epochs_nb: int,
    ctx_retrieval_lr: float,
    ctx_retrieval_dropout: float,
    cr_test_dataset_path: str,
    cr_train_dataset_path: str,
):
    print_config(_run)

    for run_i in range(runs_nb):

        cr_train_dataset = cr_dataset_from_path(cr_train_dataset_path)
        cr_test_dataset = cr_dataset_from_path(cr_test_dataset_path)

        with RunLogScope(_run, f"run{run_i}.ctx_retrieval_training"):

            ctx_retriever_model = NeuralContextRetriever.train_context_selector(
                cr_train_dataset,
                ctx_retrieval_epochs_nb,
                batch_size,
                ctx_retrieval_lr,
                _run=_run,
                log_full_loss=True,
                valid_dataset=cr_test_dataset,
                dropout=ctx_retrieval_dropout,
            )
            # NOTE: sents_nb=1 is ignored for AllContextRetriever
            ctx_retriever = NeuralContextRetriever(
                ctx_retriever_model, "all", {"sents_nb": 1}, batch_size, 1
            )
            if save_models:
                sacred_archive_huggingface_model(
                    _run, ctx_retriever_model, "ctx_retriever_model"  # type: ignore
                )

            # (len(test_ctx_retrieval), 3)
            raw_preds = ctx_retriever.predict(cr_test_dataset)

            # -1 to shift from {0, 1, 2} to {-1, 0, 1}
            preds = torch.argmax(raw_preds, dim=1).cpu() - 1

            labels = cr_test_dataset.labels()
            assert not labels is None

            # * pr curves
            #
            #   sklearn only supports binary prcurve, se we ignore
            #   the indexs where labels == -1 (or 1 for the
            #   negative pr curve)
            #
            # ** positive pr curve
            p, r, t = precision_recall_curve(
                [1 if l == 1 else 0 for l in labels], raw_preds[:, 2].cpu()
            )
            sacred_log_series(_run, "prcurve_pos_precision", p)
            sacred_log_series(_run, "prcurve_pos_recall", r)
            sacred_log_series(_run, "prcurve_pos_thresholds", t)
            # ** negative pr curve
            p, r, t = precision_recall_curve(
                [1 if l == -1 else 0 for l in labels], raw_preds[:, 0].cpu()
            )
            sacred_log_series(_run, "prcurve_neg_precision", p)
            sacred_log_series(_run, "prcurve_neg_recall", r)
            sacred_log_series(_run, "prcurve_neg_thresholds", t)

            # * micro F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="micro"
            )
            _run.log_scalar(f"precision", precision)
            _run.log_scalar(f"recall", recall)
            _run.log_scalar(f"f1", f1)
