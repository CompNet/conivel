from typing import Optional
import os
import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from conivel.datas.dekker import DekkerDataset
from conivel.datas.ontonotes import OntonotesDataset
from conivel.datas.the_hunger_games.the_hunger_games import TheHungerGamesDataset
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import RunLogScope, pretrained_bert_for_token_classification


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
    # -- common parameters
    batch_size: int
    # wether models should be saved or not
    save_models: bool = True
    # number of experiment repeats
    runs_nb: int = 5

    # -- NER training parameters
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
    ner_epochs_nb: int,
    ner_lr: float,
):
    print_config(_run)

    dekker = DekkerDataset()
    the_hunger_games = TheHungerGamesDataset(cut_into_chapters=False)

    precision_matrix = np.zeros((runs_nb,))
    recall_matrix = np.zeros((runs_nb,))
    f1_matrix = np.zeros((runs_nb,))
    metrics_matrices = [
        ("precision", precision_matrix),
        ("recall", recall_matrix),
        ("f1", f1_matrix),
    ]

    for run_i in range(runs_nb):

        with RunLogScope(_run, f"run{run_i}"):

            model = pretrained_bert_for_token_classification(
                "bert-base-cased", dekker.tag_to_id
            )
            model = train_ner_model(
                model,
                dekker,
                dekker,
                _run=_run,
                epochs_nb=ner_epochs_nb,
                batch_size=batch_size,
                learning_rate=ner_lr,
                quiet=True,
            )
            if save_models:
                sacred_archive_huggingface_model(_run, model, "model")  # type: ignore

            preds = predict(model, the_hunger_games, batch_size=batch_size).tags
            precision, recall, f1 = score_ner(
                the_hunger_games.sents(), preds, ignored_classes={"LOC", "ORG"}
            )
            _run.log_scalar(f"test_precision", precision)
            precision_matrix[run_i] = precision
            _run.log_scalar("test_recall", recall)
            recall_matrix[run_i] = recall
            _run.log_scalar("test_f1", f1)
            f1_matrix[run_i] = f1

    # global mean metrics
    for name, matrix in metrics_matrices:
        for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
            _run.log_scalar(
                f"{op_name}_test_{name}",
                op(matrix),
            )
