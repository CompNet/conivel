import os, gc, copy
from typing import List, Optional
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
import numpy as np
from conivel.datas.dataset import NERDataset
from conivel.datas.dekker import DekkerDataset
from conivel.datas.the_hunger_games import TheHungerGamesDataset
from conivel.datas.context import (
    NeuralContextRetriever,
    context_retriever_name_to_class,
)
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import (
    RunLogScope,
    sacred_archive_huggingface_model,
    sacred_archive_jsonifiable_as_file,
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
    runs_nb: int = 5

    # -- retrieval heuristic
    # pre-retrieval heuristic name
    # only officially supports 'random', 'sameword' and 'bm25' for
    # now
    retrieval_heuristic: str = "random"
    # parameters for the retrieval heuristic used when generating a
    # context retrieval dataset
    retrieval_heuristic_gen_kwargs: dict
    # parameters for the retrieval heuristic used at inference time
    retrieval_heuristic_inference_kwargs: dict

    # -- context retrieval parameters
    # number of epochs for context retrieval training
    ctx_retrieval_epochs_nb: int = 3
    # learning rate for context retrieval training
    ctx_retrieval_lr: float = 2e-5
    # usefulness threshold for context retrieval examples. Passed to
    # :func:`NeuralContextSelector.generate_context_dataset`
    ctx_retrieval_usefulness_threshold: float = 0.1
    # wether to skip examples generated from a sentence if NER
    # predictions for that sentence is correct. Passed to
    # :func:`NeuralContextSelector.generate_context_dataset`
    ctx_retrieval_skip_correct: bool = False
    # wether to use The Hunger Games dataset for context retrieval
    # dataset generation
    ctx_retrieval_dataset_generation_use_the_hunger_games: bool = False
    # number of weights bins to use to adapt the MSELoss when
    # training the neural context retriever. If ``None``, do not
    # weight the MSELoss
    ctx_retrieval_weights_bins_nb: Optional[int] = None
    # percentage of train set that will be used to train the NER model
    # used to generate the context retrieval model. The percentage
    # allocated to generate context retrieval examples will be 1 -
    # that ratio.
    ctx_retrieval_train_gen_ratio: float = 0.5

    # -- NER trainng parameters
    # list of number of sents to test
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
    book_group: Optional[str],
    folds_list: Optional[List[int]],
    batch_size: int,
    save_models: bool,
    runs_nb: int,
    retrieval_heuristic: str,
    retrieval_heuristic_gen_kwargs: dict,
    retrieval_heuristic_inference_kwargs: dict,
    ctx_retrieval_epochs_nb: int,
    ctx_retrieval_lr: float,
    ctx_retrieval_usefulness_threshold: float,
    ctx_retrieval_skip_correct: bool,
    ctx_retrieval_dataset_generation_use_the_hunger_games: bool,
    ctx_retrieval_weights_bins_nb: Optional[int],
    ctx_retrieval_train_gen_ratio: float,
    sents_nb_list: List[int],
    ner_epochs_nb: int,
    ner_lr: float,
):
    assert retrieval_heuristic in ["random", "bm25", "sameword"]
    print_config(_run)

    dekker_dataset = DekkerDataset(book_group=book_group)
    kfolds = dekker_dataset.kfolds(
        k, shuffle=not shuffle_kfolds_seed is None, shuffle_seed=shuffle_kfolds_seed
    )
    folds_nb = max(len(folds_list) if not folds_list is None else 0, len(kfolds))

    # metrics matrices
    # each matrix is of shape (runs_nb, folds_nb, sents_nb)
    # these are used to record mean metrics across folds, runs...
    precision_matrix = np.zeros((runs_nb, folds_nb, len(sents_nb_list)))
    recall_matrix = np.zeros((runs_nb, folds_nb, len(sents_nb_list)))
    f1_matrix = np.zeros((runs_nb, folds_nb, len(sents_nb_list)))
    metrics_matrices = [
        ("precision", precision_matrix),
        ("recall", recall_matrix),
        ("f1", f1_matrix),
    ]

    for run_i in range(runs_nb):

        for fold_i, (train_set, test_set) in enumerate(kfolds):

            if not folds_list is None and not fold_i in folds_list:
                continue

            # train context selector
            ner_model = pretrained_bert_for_token_classification(
                "bert-base-cased", train_set.tag_to_id
            )
            ctx_retrieval_ner_train_set, ctx_retrieval_gen_set = train_set.split(
                ctx_retrieval_train_gen_ratio
            )
            if ctx_retrieval_dataset_generation_use_the_hunger_games:
                the_hunger_games_set = TheHungerGamesDataset()
                ctx_retrieval_gen_set = NERDataset.concatenated(
                    [ctx_retrieval_gen_set, the_hunger_games_set]
                )

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.ctx_retrieval_training"):

                # train a NER model on half the training dataset
                ner_model = train_ner_model(
                    ner_model,
                    ctx_retrieval_ner_train_set,
                    ctx_retrieval_ner_train_set,
                    _run,
                    ner_epochs_nb,
                    batch_size,
                    ctx_retrieval_lr,
                )

                # generate a context retrieval dataset using the other
                # half of the training set
                ctx_retrieval_dataset = NeuralContextRetriever.generate_context_dataset(
                    ner_model,
                    ctx_retrieval_gen_set,
                    batch_size,
                    retrieval_heuristic,
                    retrieval_heuristic_gen_kwargs,
                    examples_usefulness_threshold=ctx_retrieval_usefulness_threshold,
                    skip_correct=ctx_retrieval_skip_correct,
                    _run=_run,
                )
                sacred_archive_jsonifiable_as_file(
                    _run,
                    ctx_retrieval_dataset.to_jsonifiable(),
                    "ctx_retrieval_dataset",
                )

                # NER model is not needed anymore : free the GPU of it
                del ner_model
                gc.collect()

                # train a context retriever using the previously generated
                # context retrieval dataset
                ctx_retriever_model = NeuralContextRetriever.train_context_selector(
                    ctx_retrieval_dataset,
                    ctx_retrieval_epochs_nb,
                    batch_size,
                    ctx_retrieval_lr,
                    weights_bins_nb=ctx_retrieval_weights_bins_nb,
                    _run=_run,
                )
                if save_models:
                    sacred_archive_huggingface_model(
                        _run, ctx_retriever_model, "ctx_retriever_model"  # type: ignore
                    )

            # PERFORMANCE HACK: only use the retrieval heuristic at
            # training time. At training time, the number of sentences
            # retrieved is random between ``min(sents_nb_list)`` and
            # ``max(sents_nb_list)`` for each example.
            train_set_heuristic_kwargs = copy.deepcopy(
                retrieval_heuristic_inference_kwargs
            )
            train_set_heuristic_kwargs["sents_nb"] = sents_nb_list
            train_set_heuristic = context_retriever_name_to_class[retrieval_heuristic](
                **train_set_heuristic_kwargs
            )
            ctx_train_set = train_set_heuristic(train_set)

            # train ner model on train_set
            ner_model = pretrained_bert_for_token_classification(
                "bert-base-cased", train_set.tag_to_id
            )
            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.ner"):
                ner_model = train_ner_model(
                    ner_model,
                    ctx_train_set,
                    ctx_train_set,
                    _run=_run,
                    epochs_nb=ner_epochs_nb,
                    batch_size=batch_size,
                    learning_rate=ner_lr,
                )
                if save_models:
                    sacred_archive_huggingface_model(_run, ner_model, "ner_model")  # type: ignore

            neural_context_retriever = NeuralContextRetriever(
                ctx_retriever_model,
                retrieval_heuristic,
                retrieval_heuristic_inference_kwargs,
                batch_size,
                1,
                use_cache=True,
            )

            for sents_nb_i, sents_nb in enumerate(sents_nb_list):

                _run.log_scalar("gpu_usage", gpu_memory_usage())

                neural_context_retriever.set_heuristic_sents_nb_(sents_nb)
                ctx_test_set = neural_context_retriever(test_set)

                test_preds = predict(ner_model, ctx_test_set).tags
                precision, recall, f1 = score_ner(test_set.sents(), test_preds)
                _run.log_scalar(
                    f"run{run_i}.fold{fold_i}.test_precision", precision, step=sents_nb
                )
                precision_matrix[run_i][fold_i][sents_nb_i] = precision
                _run.log_scalar(
                    f"run{run_i}.fold{fold_i}.test_recall", recall, step=sents_nb
                )
                recall_matrix[run_i][fold_i][sents_nb_i] = recall
                _run.log_scalar(f"run{run_i}.fold{fold_i}.test_f1", f1, step=sents_nb)
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
    for fold_i in range(folds_nb):
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
