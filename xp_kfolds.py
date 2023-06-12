from typing import List, Optional
import os
import numpy as np
import torch
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from conivel.datas.dataset import NERDataset
from conivel.datas.dekker import DekkerDataset, load_extended_documents
from conivel.datas.context import context_retriever_name_to_class
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
    # context retriever heuristic name
    context_retriever: str
    # context retriever extra args (not including ``sents_nb``)
    cr_kwargs = None
    # A directory containing extended documents for retrieval purposes
    # (see :meth:`.ContextRetriever.__call__)`
    cr_extended_docs_dir = None

    # -- NER training parameters
    # list of number of sents to test
    sents_nb_list: list
    # number of epochs for NER training
    ner_epochs_nb: int = 2
    # learning rate for NER training
    ner_lr: float = 2e-5
    # supplied pretrained NER models (one per fold). If None, start
    # from bert-base-cased and finetune.
    ner_model_paths: Optional[list] = None


@ex.automain
def main(
    _run: Run,
    k: int,
    shuffle_kfolds_seed: Optional[int],
    book_group: Optional[str],
    batch_size: int,
    save_models: bool,
    runs_nb: int,
    context_retriever: str,
    cr_kwargs: Optional[dict],
    cr_extended_docs_dir: Optional[str],
    sents_nb_list: List[int],
    ner_epochs_nb: int,
    ner_lr: float,
    ner_model_paths: Optional[List[str]],
):
    cr_kwargs = cr_kwargs or {}

    print_config(_run)

    dataset = DekkerDataset(book_group=book_group)
    kfolds = dataset.kfolds(
        k, shuffle=not shuffle_kfolds_seed is None, shuffle_seed=shuffle_kfolds_seed
    )

    extended_docs = None
    if cr_extended_docs_dir:
        extended_docs = load_extended_documents(cr_extended_docs_dir, dataset)

    doc_indices_map = {
        doc_attr["name"]: i for i, doc_attr in enumerate(dataset.documents_attrs)
    }
    _run.info["documents_names"] = [
        doc_attrs["name"] for doc_attrs in dataset.documents_attrs
    ]

    precision_matrix = np.zeros((runs_nb, k, len(sents_nb_list)))
    recall_matrix = np.zeros((runs_nb, k, len(sents_nb_list)))
    f1_matrix = np.zeros((runs_nb, k, len(sents_nb_list)))
    metrics_matrices = [
        ("precision", precision_matrix),
        ("recall", recall_matrix),
        ("f1", f1_matrix),
    ]

    doc_precision_matrix = np.zeros(
        (runs_nb, len(sents_nb_list), len(dataset.documents))
    )
    doc_recall_matrix = np.zeros((runs_nb, len(sents_nb_list), len(dataset.documents)))
    doc_f1_matrix = np.zeros((runs_nb, len(sents_nb_list), len(dataset.documents)))
    doc_metrics_matrices = [
        ("precision", doc_precision_matrix),
        ("recall", doc_recall_matrix),
        ("f1", doc_f1_matrix),
    ]

    for run_i in range(runs_nb):
        for fold_i, (train_set, test_set) in enumerate(kfolds):
            ctx_retriever = context_retriever_name_to_class[context_retriever](
                sents_nb=sents_nb_list, **cr_kwargs
            )
            ctx_train_set = ctx_retriever(train_set)

            # train
            if ner_model_paths is None:
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
            else:
                assert len(ner_model_paths) == k
                model = pretrained_bert_for_token_classification(
                    ner_model_paths[fold_i],
                    train_set.tag_to_id,
                )

            for sents_nb_i, sents_nb in enumerate(sents_nb_list):
                if torch.cuda.is_available():
                    _run.log_scalar("gpu_usage", gpu_memory_usage())

                ctx_retriever = context_retriever_name_to_class[context_retriever](
                    sents_nb=sents_nb, **cr_kwargs
                )

                test_preds = []

                for doc, doc_attrs in zip(test_set.documents, test_set.documents_attrs):
                    doc_i = doc_indices_map[doc_attrs["name"]]

                    ctx_doc_dataset = ctx_retriever(
                        NERDataset([doc]),
                        extended_documents=[extended_docs[doc_i]]
                        if extended_docs
                        else None,
                    )

                    doc_preds = predict(
                        model, ctx_doc_dataset, batch_size=batch_size
                    ).tags
                    test_preds += doc_preds

                    precision, recall, f1 = score_ner(
                        ctx_doc_dataset.sents(), doc_preds
                    )
                    doc_precision_matrix[run_i][sents_nb_i][doc_i] = precision
                    doc_recall_matrix[run_i][sents_nb_i][doc_i] = recall
                    doc_f1_matrix[run_i][sents_nb_i][doc_i] = f1

                precision, recall, f1 = score_ner(test_set.sents(), test_preds)
                precision_matrix[run_i][fold_i][sents_nb_i] = precision
                recall_matrix[run_i][fold_i][sents_nb_i] = recall
                f1_matrix[run_i][fold_i][sents_nb_i] = f1

        # mean metrics for the current run
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                sacred_log_series(
                    _run,
                    f"run{run_i}.{op_name}_test_{metrics_name}",
                    op(matrix[run_i], axis=0),  # (sents_nb_list)
                    steps=sents_nb_list,
                )

    # folds mean metrics
    for fold_i in range(k):
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                sacred_log_series(
                    _run,
                    f"fold{fold_i}.{op_name}_test_{metrics_name}",
                    op(matrix[:, fold_i, :], axis=0),  # (sents_nb_list)
                    steps=sents_nb_list,
                )

    # global mean metrics
    for name, matrix in metrics_matrices:
        for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
            sacred_log_series(
                _run,
                f"{op_name}_test_{name}",
                op(matrix, axis=(0, 1)),  # (sents_nb)
                steps=sents_nb_list,
            )
    for doc, doc_attrs in zip(dataset.documents, dataset.documents_attrs):
        for metrics_name, matrix in doc_metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                doc_name = doc_attrs["name"]
                doc_i = doc_indices_map[doc_name]
                sacred_log_series(
                    _run,
                    f"{op_name}_{doc_name}_test_{metrics_name}",
                    op(matrix[:, :, doc_i], axis=0),
                    steps=sents_nb_list,
                )
