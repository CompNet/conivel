import os
from typing import List, Optional
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
import numpy as np
from conivel.datas.dataset import NERDataset
from conivel.datas.dekker import DekkerDataset
from conivel.datas.context import (
    NeuralContextRetriever,
)
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import (
    RunLogScope,
    sacred_archive_huggingface_model,
    sacred_log_series,
    gpu_memory_usage,
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
    test_books = list

    # -- common parameters
    batch_size: int
    # wether models should be saved or not
    save_models: bool = True
    # number of experiment repeats
    runs_nb: int = 5

    # -- retrieval heuristic
    # pre-retrieval heuristic name
    retrieval_heuristic: str = "random"
    retrieval_heuristic_kwargs: dict
    neural_retriever_path: str

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
    test_books: List[str],
    batch_size: int,
    save_models: bool,
    runs_nb: int,
    retrieval_heuristic: str,
    retrieval_heuristic_kwargs: dict,
    neural_retriever_path: str,
    sents_nb_list: List[int],
    ner_epochs_nb: int,
    ner_lr: float,
):
    print_config(_run)

    dataset = DekkerDataset()
    test = []
    train = []
    for document_name, document in zip(dataset.documents_names, dataset.documents):
        if document_name in test_books:
            test.append(document)
        else:
            train.append(document)
    test = NERDataset(test, dataset.tags, dataset.tokenizer)
    train = NERDataset(train, dataset.tags, dataset.tokenizer)

    # metrics matrices
    # each matrix is of shape (runs_nb, folds_nb, sents_nb)
    # these are used to record mean metrics across folds, runs...
    precision_matrix = np.zeros((runs_nb, len(sents_nb_list)))
    recall_matrix = np.zeros((runs_nb, len(sents_nb_list)))
    f1_matrix = np.zeros((runs_nb, len(sents_nb_list)))
    metrics_matrices = [
        ("precision", precision_matrix),
        ("recall", recall_matrix),
        ("f1", f1_matrix),
    ]

    for run_i in range(runs_nb):

        neural_retriever = NeuralContextRetriever(
            neural_retriever_path,
            retrieval_heuristic,
            retrieval_heuristic_kwargs,
            batch_size,
            1,
        )
        ctx_train_set = neural_retriever(train, quiet=False)

        # train ner model on train_set
        ner_model = pretrained_bert_for_token_classification(
            "bert-base-cased", train.tag_to_id
        )
        with RunLogScope(_run, f"run{run_i}.ner"):
            ner_model = train_ner_model(
                ner_model,
                ctx_train_set,
                ctx_train_set,
                _run=_run,
                epochs_nb=ner_epochs_nb,
                batch_size=batch_size,
                learning_rate=ner_lr,
                quiet=False,
            )
            if save_models:
                sacred_archive_huggingface_model(_run, ner_model, "ner_model")  # type: ignore

        for sents_nb_i, sents_nb in enumerate(sents_nb_list):

            _run.log_scalar("gpu_usage", gpu_memory_usage())

            neural_retriever.sents_nb = sents_nb
            ctx_test_set = neural_retriever(test, quiet=False)

            test_preds = predict(ner_model, ctx_test_set, quiet=True).tags
            precision, recall, f1 = score_ner(test.sents(), test_preds)
            _run.log_scalar(f"run{run_i}.test_precision", precision, step=sents_nb)
            precision_matrix[run_i][sents_nb_i] = precision
            _run.log_scalar(f"run{run_i}.test_recall", recall, step=sents_nb)
            recall_matrix[run_i][sents_nb_i] = recall
            _run.log_scalar(f"run{run_i}.test_f1", f1, step=sents_nb)
            f1_matrix[run_i][sents_nb_i] = f1

    # global mean metrics
    for name, matrix in metrics_matrices:
        for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
            sacred_log_series(
                _run,
                f"{op_name}_test_{name}",
                op(matrix, axis=0),  # (sents_nb)
                steps=sents_nb_list,
            )
