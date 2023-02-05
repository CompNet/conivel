from typing import List
import os
import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from conivel.datas.dekker import DekkerDataset
from conivel.datas.ontonotes import OntonotesDataset
from conivel.datas.context import context_retriever_name_to_class
from conivel.datas.the_hunger_games.the_hunger_games import TheHungerGamesDataset
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import (
    RunLogScope,
    gpu_memory_usage,
    sacred_archive_huggingface_model,
    sacred_log_series,
    pretrained_bert_for_token_classification,
)


script_dir = os.path.abspath(os.path.dirname(__file__))

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))
if os.path.isfile(f"{script_dir}/telegram_observer_config.json"):
    ex.observers.append(
        TelegramObserver.from_config(f"{script_dir}/telegram_observer_config.json")
    )


@ex.config
def config():
    # -- datas parameters
    # -- common parameters
    batch_size: int
    # wether models should be saved or not
    save_models: bool = True
    # number of experiment repeats
    runs_nb: int = 5

    # -- context retrieval
    # context retriever heuristic name
    context_retriever: str
    # context retriever extra args (not including ``sents_nb``)
    context_retriever_kwargs: dict

    # -- NER training parameters
    # list of number of sents to test
    sents_nb_list: list
    # number of epochs for NER training
    ner_epochs_nb: int = 2
    # learning rate for NER training
    ner_lr: float = 2e-5


@ex.automain
def main(
    _run: Run,
    batch_size: int,
    save_models: bool,
    runs_nb: int,
    context_retriever: str,
    context_retriever_kwargs: dict,
    sents_nb_list: List[int],
    ner_epochs_nb: int,
    ner_lr: float,
):
    print_config(_run)

    dekker = DekkerDataset()
    the_hunger_games = TheHungerGamesDataset()

    precision_matrix = np.zeros((runs_nb, len(sents_nb_list)))
    recall_matrix = np.zeros((runs_nb, len(sents_nb_list)))
    f1_matrix = np.zeros((runs_nb, len(sents_nb_list)))
    metrics_matrices = [
        ("precision", precision_matrix),
        ("recall", recall_matrix),
        ("f1", f1_matrix),
    ]

    for run_i in range(runs_nb):

        ctx_retriever = context_retriever_name_to_class[context_retriever](
            sents_nb=sents_nb_list, **context_retriever_kwargs
        )
        ctx_train_set = ctx_retriever(dekker)

        # train
        with RunLogScope(_run, f"run{run_i}"):

            model = pretrained_bert_for_token_classification(
                "bert-base-cased", ctx_train_set.tag_to_id
            )
            model = train_ner_model(
                model,
                ctx_train_set,
                ctx_train_set,
                _run=_run,
                epochs_nb=ner_epochs_nb,
                batch_size=batch_size,
                learning_rate=ner_lr,
                quiet=True,
            )
            if save_models:
                sacred_archive_huggingface_model(_run, model, "model")  # type: ignore

        for sents_nb_i, sents_nb in enumerate(sents_nb_list):

            _run.log_scalar("gpu_usage", gpu_memory_usage())

            ctx_retriever = context_retriever_name_to_class[context_retriever](
                sents_nb=sents_nb, **context_retriever_kwargs
            )
            ctx_test_set = ctx_retriever(the_hunger_games)

            # test
            test_preds = predict(model, ctx_test_set, batch_size=batch_size).tags
            precision, recall, f1 = score_ner(ctx_test_set.sents(), test_preds)
            _run.log_scalar(
                f"run{run_i}.test_precision",
                precision,
                step=sents_nb,
            )
            precision_matrix[run_i][sents_nb_i] = precision
            _run.log_scalar(f"run{run_i}.test_recall", recall, step=sents_nb)
            recall_matrix[run_i][sents_nb_i] = recall
            _run.log_scalar(f"run{run_i}.test_f1", f1, step=sents_nb)
            f1_matrix[run_i][sents_nb_i] = f1

        # mean metrics for the current run
        for metrics_name, matrix in metrics_matrices:
            sacred_log_series(
                _run,
                f"run{run_i}.mean_test_{metrics_name}",
                matrix[run_i],
                steps=sents_nb_list,
            )

    # global mean metrics
    for name, matrix in metrics_matrices:
        for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
            sacred_log_series(
                _run,
                f"{op_name}_test_{name}",
                op(matrix, axis=0),  # (sents_nb)
                steps=sents_nb_list,
            )
