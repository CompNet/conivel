from typing import List, Optional, Literal
import os
import numpy as np
from tqdm import tqdm
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset
from conivel.datas.dekker import DekkerDataset
from conivel.datas.ontonotes import OntonotesDataset
from conivel.datas.context import ContextRetriever, context_retriever_name_to_class
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import (
    flattened,
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


def call_retrievers(
    retrievers: List[ContextRetriever], dataset: NERDataset
) -> NERDataset:
    """Use a combination of several retrievers to retrieve context."""
    new_docs = []
    for document in tqdm(dataset.documents):
        new_doc = []
        for sent_i, sent in enumerate(document):
            retrieval_matchs = flattened(
                [r.retrieve(sent_i, document) for r in retrievers]
            )
            # unique matchs based on their sentence index
            retrieval_matchs = list(
                {rmatch.sentence_idx: rmatch for rmatch in retrieval_matchs}.values()
            )
            # sort by position in text
            retrieval_matchs = sorted(retrieval_matchs, key=lambda m: m.sentence_idx)
            new_doc.append(
                NERSentence(
                    sent.tokens,
                    sent.tags,
                    [m.sentence for m in retrieval_matchs if m.side == "left"],
                    [m.sentence for m in retrieval_matchs if m.side == "right"],
                )
            )
        new_docs.append(new_doc)
    return NERDataset(new_docs, tags=dataset.tags, tokenizer=dataset.tokenizer)


@ex.config
def config():
    # -- datas parameters
    # number of folds
    k: int = 5
    # seed to use when folds shuffling. If ``None``, no shuffling is
    # performed.
    shuffle_kfolds_seed: Optional[int] = None

    # -- common parameters
    batch_size: int
    # wether models should be saved or not
    save_models: bool = True
    # number of experiment repeats
    runs_nb: int = 5

    # -- context retrieval
    # context retriever heuristic names
    retrievers_names: list

    # -- NER training parameters
    # list of number of sents to test _per retriever_
    sents_nb_list: list
    # number of epochs for NER training
    ner_epochs_nb: int = 2
    # learning rate for NER training
    ner_lr: float = 2e-5


@ex.automain
def main(
    _run: Run,
    k: int,
    shuffle_kfolds_seed: Optional[int],
    batch_size: int,
    save_models: bool,
    runs_nb: int,
    retrievers_names: List[str],
    sents_nb_list: List[int],
    ner_epochs_nb: int,
    ner_lr: float,
):
    print_config(_run)

    dataset = DekkerDataset()
    kfolds = dataset.kfolds(
        k, shuffle=not shuffle_kfolds_seed is None, shuffle_seed=shuffle_kfolds_seed
    )

    precision_matrix = np.zeros((runs_nb, k, len(sents_nb_list)))
    recall_matrix = np.zeros((runs_nb, k, len(sents_nb_list)))
    f1_matrix = np.zeros((runs_nb, k, len(sents_nb_list)))
    metrics_matrices = [
        ("precision", precision_matrix),
        ("recall", recall_matrix),
        ("f1", f1_matrix),
    ]

    for run_i in range(runs_nb):

        for fold_i, (train_set, test_set) in enumerate(kfolds):

            # * context retrievers instantiation
            ctx_retrievers = [
                context_retriever_name_to_class[r](sents_nb_list)
                for r in retrievers_names
            ]

            # * context retrieval
            train_set = call_retrievers(ctx_retrievers, train_set)

            # * training
            with RunLogScope(_run, f"run{run_i}.fold{fold_i}"):

                model = pretrained_bert_for_token_classification(
                    "bert-base-cased", train_set.tag_to_id
                )
                model = train_ner_model(
                    model,
                    train_set,
                    train_set,
                    _run=_run,
                    epochs_nb=ner_epochs_nb,
                    batch_size=batch_size,
                    learning_rate=ner_lr,
                )
                if save_models:
                    sacred_archive_huggingface_model(_run, model, "model")  # type: ignore

            # * overall testing
            for sents_nb_i, sents_nb in enumerate(sents_nb_list):

                _run.log_scalar("gpu_usage", gpu_memory_usage())

                # * context retriever settings
                for retriever in ctx_retrievers:
                    retriever.sents_nb = sents_nb

                # * context retrieval
                ctx_test_set = call_retrievers(ctx_retrievers, test_set)

                # * test predictions
                test_preds = predict(model, ctx_test_set, batch_size=batch_size).tags

                # * test scoring
                precision, recall, f1 = score_ner(ctx_test_set.sents(), test_preds)

                # * metrics logging
                _run.log_scalar(
                    f"run{run_i}.fold{fold_i}.test_precision",
                    precision,
                    step=sents_nb,
                )
                precision_matrix[run_i][fold_i][sents_nb_i] = precision
                _run.log_scalar(
                    f"run{run_i}.fold{fold_i}.test_recall", recall, step=sents_nb
                )
                recall_matrix[run_i][fold_i][sents_nb_i] = recall
                _run.log_scalar(f"run{run_i}.fold{fold_i}.test_f1", f1, step=sents_nb)
                f1_matrix[run_i][fold_i][sents_nb_i] = f1

        # * mean metrics for the current run
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                sacred_log_series(
                    _run,
                    f"run{run_i}.{op_name}_test_{metrics_name}",
                    op(matrix[run_i], axis=0),  # (sents_nb_list)
                    steps=sents_nb_list,
                )

    # * folds mean metrics
    for fold_i in range(k):
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                sacred_log_series(
                    _run,
                    f"fold{fold_i}.{op_name}_test_{metrics_name}",
                    op(matrix[:, fold_i, :], axis=0),  # (sents_nb_list)
                    steps=sents_nb_list,
                )

    # * global mean metrics
    for name, matrix in metrics_matrices:
        for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
            sacred_log_series(
                _run,
                f"{op_name}_test_{name}",
                op(matrix, axis=(0, 1)),  # (sents_nb)
                steps=sents_nb_list,
            )
