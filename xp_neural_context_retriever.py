import os
from typing import List, Literal, Optional, Tuple
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from FastChat.fastchat.serve.inference import load_model
from conivel.datas.dataset import NERDataset
from conivel.datas.dekker import DekkerDataset
from conivel.datas.context import (
    ContextRetrievalDataset,
    NeuralContextRetriever,
)
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import (
    RunLogScope,
    sacred_archive_huggingface_model,
    sacred_archive_jsonifiable_as_file,
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


def gen_cr_dataset(dataset: NERDataset) -> ContextRetrievalDataset:
    llm_model = load_model("TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g", "cpu", 0)
    # TODO:


@ex.config
def config():
    # -- datas parameters
    # number of folds
    k: int = 5
    # seed to use when shuffling folds. If ``None``, no shuffling is
    # performed.
    shuffle_kfolds_seed: Optional[int] = None

    # -- common parameters
    batch_size: int
    # wether models should be saved or not
    save_models: bool = True
    # number of experiment repeats
    runs_nb: int

    # -- context retrieval parameters
    # number of epochs for context retrieval training
    cr_epochs_nb: int = 3
    # learning rate for context retrieval training
    cr_lr: float = 2e-5

    # -- NER parameters
    ner_epochs_nb: int = 2
    ner_lr: float = 2e-5


@ex.automain
def main(
    _run: Run,
    k: int,
    shuffle_kfolds_seed: Optional[int],
    batch_size: int,
    save_models: bool,
    runs_nb: int,
    cr_epochs_nb: int,
    cr_lr: float,
    cr_dropout: float,
    ner_epochs_nb: int,
    ner_lr: float,
):
    print_config(_run)

    dataset = DekkerDataset()
    kfolds = dataset.kfolds(
        k, shuffle=not shuffle_kfolds_seed is None, shuffle_seed=shuffle_kfolds_seed
    )
    folds_nb = len(kfolds)

    # * Metrics matrices
    #   each matrix is of shape (runs_nb, folds_nb, sents_nb)
    #   these are used to record mean metrics across folds, runs...
    cr_precision_matrix = np.zeros((runs_nb, folds_nb))
    cr_recall_matrix = np.zeros((runs_nb, folds_nb))
    cr_f1_matrix = np.zeros((runs_nb, folds_nb))
    ner_precision_matrix = np.zeros((runs_nb, folds_nb))
    ner_recall_matrix = np.zeros((runs_nb, folds_nb))
    ner_f1_matrix = np.zeros((runs_nb, folds_nb))
    metrics_matrices: List[Tuple[str, np.ndarray]] = [
        ("cr_precision", cr_precision_matrix),
        ("cr_recall", cr_recall_matrix),
        ("cr_f1", cr_f1_matrix),
        ("ner_precision", ner_precision_matrix),
        ("ner_recall", ner_recall_matrix),
        ("ner_f1", ner_f1_matrix),
    ]

    for run_i in range(runs_nb):

        for fold_i, (train, test) in enumerate(kfolds):

            _run.log_scalar("gpu_usage", gpu_memory_usage())

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.cr_dataset_generation"):

                cr_train_dataset = gen_cr_dataset(train)
                sacred_archive_jsonifiable_as_file(
                    _run, cr_train_dataset.to_jsonifiable(), "cr_train_dataset"
                )

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.cr_model_training"):

                neural_retriever_model = NeuralContextRetriever.train_context_selector(
                    cr_train_dataset,
                    cr_epochs_nb,
                    batch_size,
                    cr_lr,
                    _run=_run,
                    log_full_loss=True,
                    dropout=cr_dropout,
                )
                neural_retriever = NeuralContextRetriever(
                    neural_retriever_model,
                    "all",
                    {"sents_nb": 1},  # WARNING: ignored
                    batch_size,
                    1,
                )
                if save_models:
                    sacred_archive_huggingface_model(
                        _run, neural_retriever_model, "cr_model"  # type: ignore
                    )

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.cr_model_testing"):

                cr_test_dataset = gen_cr_dataset(test)
                sacred_archive_jsonifiable_as_file(
                    _run, cr_test_dataset.to_jsonifiable(), "cr_test_dataset"
                )

                # (len(test_ctx_retrieval), 2)
                raw_preds = neural_retriever.predict(cr_test_dataset)
                preds = torch.argmax(raw_preds, dim=1).cpu()
                labels = cr_test_dataset.labels()
                assert not labels is None

                # * Micro F1
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, preds, average="micro"
                )

                cr_precision_matrix[run_i][fold_i] = precision
                cr_recall_matrix[run_i][fold_i] = recall
                cr_f1_matrix[run_i][fold_i] = f1
                _run.log_scalar(f"precision", precision)
                _run.log_scalar(f"recall", recall)
                _run.log_scalar(f"f1", f1)

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.ner_model_training"):

                train_and_ctx = neural_retriever(train, quiet=False)
                ner_model = pretrained_bert_for_token_classification(
                    "bert-base-cased", train.tag_to_id
                )
                ner_model = train_ner_model(
                    ner_model,
                    train_and_ctx,
                    train_and_ctx,
                    _run,
                    epochs_nb=ner_epochs_nb,
                    batch_size=batch_size,
                    learning_rate=ner_lr,
                )

                if save_models:
                    sacred_archive_huggingface_model(_run, ner_model, f"ner_model")

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.ner_model_testing"):

                test_and_ctx = neural_retriever(test, quiet=False)
                preds = predict(ner_model, test_and_ctx, batch_size=batch_size).tags
                precision, recall, f1 = score_ner(test.sents(), preds)

                ner_precision_matrix[run_i][fold_i] = precision
                ner_recall_matrix[run_i][fold_i] = recall
                ner_f1_matrix[run_i][fold_i] = f1
                _run.log_scalar("precision", precision)
                _run.log_scalar(f"recall", recall)
                _run.log_scalar(f"f1", f1)

        # * Run mean metrics
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                _run.log_scalar(
                    f"run{run_i}.{op_name}_test_{metrics_name}",
                    op(matrix[run_i], axis=0),
                )

    # * Folds mean metrics
    for fold_i in range(folds_nb):
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                _run.log_scalar(
                    f"fold{fold_i}.{op_name}_test_{metrics_name}",
                    op(matrix[:, fold_i], axis=0),
                )

    # * Global mean metrics
    for name, matrix in metrics_matrices:
        for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
            _run.log_scalar(
                f"{op_name}_test_{name}",
                op(matrix, axis=(0, 1)),
            )
