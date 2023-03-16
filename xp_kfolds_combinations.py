from typing import List, Optional, Literal
import os
import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from conivel.datas.dekker import DekkerDataset
from conivel.datas.ontonotes import OntonotesDataset
from conivel.datas.context import (
    PipelineContextRetriever,
    context_retriever_name_to_class,
)
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
    # number of folds
    k: int = 5
    # seed to use when folds shuffling. If ``None``, no shuffling is
    # performed.
    shuffle_kfolds_seed: Optional[int] = None
    # wether to restrict the experiment to a group of book in the
    # Dekker et al's dataset
    book_group: Optional[str] = None

    # -- common parameters
    batch_size: int
    # wether models should be saved or not
    save_models: bool = True
    # number of experiment repeats
    runs_nb: int = 5

    # -- context retrieval
    # list of context retrievers heuristics. Each context retriever
    # will be applied after the other.
    context_retrievers: list
    # list of extra context retriever args (not including ``sents_nb``)
    context_retrievers_kwargs: list
    # the max number of sents to retrieve for each context retriever
    context_retrievers_sents_nb: list

    # -- NER training parameters
    # number of epochs for NER training
    ner_epochs_nb: int = 2
    # learning rate for NER training
    ner_lr: float = 2e-5

    # --
    # one of : 'dekker', 'ontonotes'
    dataset_name: str = "dekker"
    # if dataset_name == 'ontonotes'
    dataset_path: Optional[str] = None


@ex.automain
def main(
    _run: Run,
    k: int,
    shuffle_kfolds_seed: Optional[int],
    book_group: Optional[str],
    batch_size: int,
    save_models: bool,
    runs_nb: int,
    context_retrievers: List[str],
    context_retrievers_kwargs: List[dict],
    context_retrievers_sents_nb: List[int],
    ner_epochs_nb: int,
    ner_lr: float,
    dataset_name: Literal["dekker", "ontonotes"],
    dataset_path: Optional[str],
):
    print_config(_run)

    if dataset_name == "dekker":
        dataset = DekkerDataset(book_group=book_group)
    elif dataset_name == "ontonotes":
        assert not dataset_path is None
        dataset = OntonotesDataset(dataset_path)
        # keep only documents with a number of tokens >= 512
        dataset.documents = [
            doc for doc in dataset.documents if sum([len(sent) for sent in doc]) >= 512
        ]
    else:
        raise ValueError(f"unknown dataset name {dataset_name}")
    kfolds = dataset.kfolds(
        k, shuffle=not shuffle_kfolds_seed is None, shuffle_seed=shuffle_kfolds_seed
    )

    precision_matrix = np.zeros((runs_nb, k))
    recall_matrix = np.zeros((runs_nb, k))
    f1_matrix = np.zeros((runs_nb, k))
    metrics_matrices = [
        ("precision", precision_matrix),
        ("recall", recall_matrix),
        ("f1", f1_matrix),
    ]

    for run_i in range(runs_nb):

        for fold_i, (train_set, test_set) in enumerate(kfolds):

            ctx_retrievers = [
                context_retriever_name_to_class[retriever](
                    sents_nb=context_retrievers_sents_nb[i],
                    **context_retrievers_kwargs[i],
                )
                for i, retriever in enumerate(context_retrievers)
            ]
            pipeline_retriever = PipelineContextRetriever(None, ctx_retrievers)
            ctx_train_set = pipeline_retriever(train_set)

            # train
            with RunLogScope(_run, f"run{run_i}.fold{fold_i}"):

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

            _run.log_scalar("gpu_usage", gpu_memory_usage())

            ctx_test_set = pipeline_retriever(test_set)

            # test
            test_preds = predict(model, ctx_test_set, batch_size=batch_size).tags
            precision, recall, f1 = score_ner(ctx_test_set.sents(), test_preds)
            _run.log_scalar(
                f"run{run_i}.fold{fold_i}.test_precision",
                precision,
            )
            precision_matrix[run_i][fold_i] = precision
            _run.log_scalar(f"run{run_i}.fold{fold_i}.test_recall", recall)
            recall_matrix[run_i][fold_i] = recall
            _run.log_scalar(f"run{run_i}.fold{fold_i}.test_f1", f1)
            f1_matrix[run_i][fold_i] = f1

        # mean metrics for the current run
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                _run.log_scalar(
                    f"run{run_i}.{op_name}_test_{metrics_name}", op(matrix[run_i])
                )

    # folds mean metrics
    for fold_i in range(k):
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                _run.log_scalar(
                    f"fold{fold_i}.{op_name}_test_{metrics_name}", op(matrix[:, fold_i])
                )

    # global mean metrics
    for name, matrix in metrics_matrices:
        for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
            _run.log_scalar(f"{op_name}_test_{name}", op(matrix))
