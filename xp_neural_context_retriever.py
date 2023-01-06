import os
from typing import List, Literal, Optional, Tuple
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
)
from conivel.datas.ontonotes import OntonotesDataset
from conivel.datas.dekker import DekkerDataset
from conivel.datas.context import (
    NeuralContextRetriever,
)
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
    # list of folds number (starting from 0) to perform the experiment
    # on. If not specified, perform the experiment on all folds
    folds_list: Optional[list] = None

    # -- common parameters
    batch_size: int
    # wether models should be saved or not
    save_models: bool = True
    # number of experiment repeats
    runs_nb: int

    # -- retrieval heuristic
    # pre-retrieval heuristic name
    # only officially supports 'random', 'samenoun' and 'bm25' for
    # now
    retrieval_heuristic: str = "random"
    # parameters for the retrieval heuristic used when generating a
    # context retrieval dataset
    retrieval_heuristic_gen_kwargs: dict

    # -- context retrieval parameters
    # number of epochs for context retrieval training
    ctx_retrieval_epochs_nb: int = 3
    # learning rate for context retrieval training
    ctx_retrieval_lr: float = 2e-5
    # percentage of train set that will be used to train the NER model
    # used to generate the context retrieval model. The percentage
    # allocated to generate context retrieval examples will be 1 -
    # that ratio.
    ctx_retrieval_train_gen_ratio: float = 0.5
    # downsampling ratio for the examples that have no impact on
    # predictions
    ctx_retrieval_downsampling_ratio: float = 0.05

    # --
    # one of : 'dekker', 'ontonotes'
    dataset_name: str = "dekker"
    # if dataset_name == 'ontonotes
    dataset_path: Optional[str] = None


@ex.automain
def main(
    _run: Run,
    k: int,
    shuffle_kfolds_seed: Optional[int],
    book_group: Optional[str],
    folds_list: Optional[List[int]],
    batch_size: int,
    save_models: bool,
    runs_nb: int,
    retrieval_heuristic: str,
    retrieval_heuristic_gen_kwargs: dict,
    ctx_retrieval_epochs_nb: int,
    ctx_retrieval_lr: float,
    ctx_retrieval_train_gen_ratio: float,
    ctx_retrieval_downsampling_ratio: float,
    dataset_name: Literal["dekker", "ontonotes"],
    dataset_path: Optional[str],
):
    assert retrieval_heuristic in ["random", "bm25", "samenoun"]
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
    folds_nb = max(len(folds_list) if not folds_list is None else 0, len(kfolds))

    # metrics matrices
    # each matrix is of shape (runs_nb, folds_nb, sents_nb)
    # these are used to record mean metrics across folds, runs...
    precision_matrix = np.zeros((runs_nb, folds_nb))
    recall_matrix = np.zeros((runs_nb, folds_nb))
    f1_matrix = np.zeros((runs_nb, folds_nb))
    pos_precision_matrix = np.zeros((runs_nb, folds_nb))
    pos_recall_matrix = np.zeros((runs_nb, folds_nb))
    pos_f1_matrix = np.zeros((runs_nb, folds_nb))
    metrics_matrices: List[Tuple[str, np.ndarray]] = [
        ("precision", precision_matrix),
        ("recall", recall_matrix),
        ("f1", f1_matrix),
        ("pos_precision", pos_precision_matrix),
        ("pos_recall", pos_recall_matrix),
        ("pos_f1", pos_f1_matrix),
    ]

    for run_i in range(runs_nb):

        for fold_i, (train_set, test_set) in enumerate(kfolds):

            _run.log_scalar("gpu_usage", gpu_memory_usage())

            if not folds_list is None and not fold_i in folds_list:
                continue

            # train context selector
            ner_model = pretrained_bert_for_token_classification(
                "bert-base-cased", train_set.tag_to_id
            )

            ctx_retrieval_ner_train_set, ctx_retrieval_gen_set = train_set.split(
                ctx_retrieval_train_gen_ratio
            )

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.ctx_retrieval_training"):

                # train a NER model on half the training dataset
                ner_model = train_ner_model(
                    ner_model,
                    ctx_retrieval_ner_train_set,
                    ctx_retrieval_ner_train_set,
                    _run=_run,
                    epochs_nb=1,  # only 1 epoch training to allow the model to make mistakes
                    batch_size=batch_size,
                    learning_rate=ctx_retrieval_lr,
                )

                # generate a context retrieval dataset using the other
                # half of the training set
                ctx_retrieval_dataset = NeuralContextRetriever.generate_context_dataset(
                    ner_model,
                    ctx_retrieval_gen_set,
                    batch_size,
                    retrieval_heuristic,
                    retrieval_heuristic_gen_kwargs,
                    _run=_run,
                )
                # downsample the majority class (0) of the dataset
                ctx_retrieval_dataset = ctx_retrieval_dataset.downsampled(
                    ctx_retrieval_downsampling_ratio
                )
                # save dataset
                sacred_archive_jsonifiable_as_file(
                    _run,
                    ctx_retrieval_dataset.to_jsonifiable(),
                    "ctx_retrieval_dataset",
                )

                # train a context retriever using the previously generated
                # context retrieval dataset
                # weights = torch.tensor(
                #     [
                #         1 / ctx_retrieval_downsampling_ratio,
                #         1.0,
                #         1 / ctx_retrieval_downsampling_ratio,
                #     ]
                # )
                weights = None
                ctx_retriever_model = NeuralContextRetriever.train_context_selector(
                    ctx_retrieval_dataset,
                    ctx_retrieval_epochs_nb,
                    batch_size,
                    ctx_retrieval_lr,
                    _run=_run,
                    weights=weights,
                    log_full_loss=True,
                )
                ctx_retriever = NeuralContextRetriever(
                    ctx_retriever_model,
                    retrieval_heuristic,
                    retrieval_heuristic_gen_kwargs,
                    batch_size,
                    1,
                )
                if save_models:
                    sacred_archive_huggingface_model(
                        _run, ctx_retriever_model, "ctx_retriever_model"  # type: ignore
                    )

                test_ctx_retrieval_dataset = (
                    NeuralContextRetriever.generate_context_dataset(
                        ner_model,
                        test_set,
                        batch_size,
                        retrieval_heuristic,
                        {"sents_nb": 1},
                        _run=_run,
                    )
                )
                preds = ctx_retriever.predict(test_ctx_retrieval_dataset)
                # -1 to shift from {0, 1, 2} to {-1, 0, 1}
                preds = torch.argmax(preds, dim=1).cpu() - 1

            labels = test_ctx_retrieval_dataset.labels()
            assert not labels is None

            # micro F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="micro"
            )
            precision_matrix[run_i][fold_i] = precision
            recall_matrix[run_i][fold_i] = recall
            f1_matrix[run_i][fold_i] = f1
            _run.log_scalar(f"run{run_i}.fold{fold_i}.precision", precision)
            _run.log_scalar(f"run{run_i}.fold{fold_i}.recall", recall)
            _run.log_scalar(f"run{run_i}.fold{fold_i}.f1", f1)

            # record precision, recall and f1 of the positive class
            precisions, recalls, f1s, _ = precision_recall_fscore_support(labels, preds)
            pos_precision_matrix[run_i][fold_i] = precisions[2]
            pos_recall_matrix[run_i][fold_i] = recalls[2]
            pos_f1_matrix[run_i][fold_i] = f1s[2]
            _run.log_scalar(f"run{run_i}.fold{fold_i}.pos_precision", precisions[2])
            _run.log_scalar(f"run{run_i}.fold{fold_i}.pos_recall", recalls[2])
            _run.log_scalar(f"run{run_i}.fold{fold_i}.pos_f1", f1s[2])

        # mean metrics for the current run
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                _run.log_scalar(
                    f"run{run_i}.{op_name}_test_{metrics_name}",
                    op(matrix[run_i], axis=0),
                )

    # folds mean metrics
    for fold_i in range(folds_nb):
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                _run.log_scalar(
                    f"fold{fold_i}.{op_name}_test_{metrics_name}",
                    op(matrix[:, fold_i], axis=0),
                )

    # global mean metrics
    for name, matrix in metrics_matrices:
        for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
            _run.log_scalar(
                f"{op_name}_test_{name}",
                op(matrix, axis=(0, 1)),
            )
