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
from conivel.datas.context import (
    MonoBERTContextRetriever,
    CombinedContextRetriever,
    RandomContextRetriever,
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
    sacred_archive_jsonifiable_as_file,
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
    # context retriever heuristics names
    cr_heuristics: list
    # kwargs for the context retrieval heuristics
    cr_heuristics_kwargs: list
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
    # Huggingface ID of the NER model to finetune
    ner_model_id: str = "bert-base-cased"
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
    cr_heuristics: List[str],
    cr_heuristics_kwargs: List[dict],
    cr_extended_docs_dir: Optional[str],
    sents_nb_list: List[int],
    ner_epochs_nb: int,
    ner_lr: float,
    ner_model_id: str,
    ner_model_paths: Optional[List[str]],
):
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

            # train
            if ner_model_paths is None:
                with RunLogScope(_run, f"run{run_i}.fold{fold_i}"):
                    random_retriever = RandomContextRetriever(1)
                    ctx_train_set = random_retriever(train_set, quiet=False)
                    model = pretrained_bert_for_token_classification(
                        ner_model_id, ctx_train_set.tag_to_id
                    )
                    model = train_ner_model(
                        model,
                        ctx_train_set,
                        ctx_train_set,
                        _run=_run,
                        epochs_nb=ner_epochs_nb,
                        batch_size=batch_size,
                        learning_rate=ner_lr,
                        quiet=False,
                    )
                    if save_models:
                        sacred_archive_huggingface_model(_run, model, "model")  # type: ignore
            else:
                assert len(ner_model_paths) == k
                model = pretrained_bert_for_token_classification(
                    ner_model_paths[fold_i],
                    train_set.tag_to_id,
                )

            monobert_retriever = MonoBERTContextRetriever(
                max(sents_nb_list),
                CombinedContextRetriever(
                    sum([kw["sents_nb"] for kw in cr_heuristics_kwargs]),
                    [
                        context_retriever_name_to_class[name](**kw)
                        for name, kw in zip(cr_heuristics, cr_heuristics_kwargs)
                    ],
                ),
            )

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.ner_model_testing"):
                test_preds = [[] for _ in range(len(sents_nb_list))]

                for doc, doc_attrs in zip(test_set.documents, test_set.documents_attrs):
                    doc_name = doc_attrs["name"]
                    doc_i = doc_indices_map[doc_attrs["name"]]
                    doc_dataset = NERDataset([doc], documents_attrs=[doc_attrs])

                    if torch.cuda.is_available():
                        _run.log_scalar("gpu_usage", gpu_memory_usage())

                    for sents_nb_i, ctx_doc_dataset in enumerate(
                        monobert_retriever.dataset_with_contexts(
                            doc_dataset,
                            sents_nb_list,
                            quiet=False,
                            extended_documents=[extended_docs[doc_i]]
                            if extended_docs
                            else None,
                        )
                    ):

                        sacred_archive_jsonifiable_as_file(
                            _run,
                            [s.to_jsonifiable() for s in ctx_doc_dataset.sents()],
                            f"{doc_name}_retrieved_dataset",
                        )

                        doc_preds = predict(
                            model, ctx_doc_dataset, batch_size=batch_size
                        ).tags
                        test_preds[sents_nb_i] += doc_preds

                        precision, recall, f1 = score_ner(
                            doc_dataset.sents(), doc_preds
                        )
                        doc_precision_matrix[run_i][sents_nb_i][doc_i] = precision
                        doc_recall_matrix[run_i][sents_nb_i][doc_i] = recall
                        doc_f1_matrix[run_i][sents_nb_i][doc_i] = f1

                for sents_nb_i, sents_nb_test_preds in enumerate(test_preds):
                    precision, recall, f1 = score_ner(
                        test_set.sents(), sents_nb_test_preds
                    )
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
