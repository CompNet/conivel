import os, gc, copy
from typing import List, Optional
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
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
    # inference sents_nb to test
    retrieval_heuristic_inference_sents_nb_list: list

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

    # -- NER trainng parameters
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
    retrieval_heuristic_inference_sents_nb_list: List[int],
    ctx_retrieval_epochs_nb: int,
    ctx_retrieval_lr: float,
    ctx_retrieval_usefulness_threshold: float,
    ctx_retrieval_skip_correct: bool,
    ctx_retrieval_dataset_generation_use_the_hunger_games: bool,
    ctx_retrieval_weights_bins_nb: Optional[int],
    ner_epochs_nb: int,
    ner_lr: float,
):
    assert retrieval_heuristic in ["random", "bm25", "sameword"]
    print_config(_run)

    dekker_dataset = DekkerDataset(book_group=book_group)
    kfolds = dekker_dataset.kfolds(
        k, shuffle=not shuffle_kfolds_seed is None, shuffle_seed=shuffle_kfolds_seed
    )

    for run_i in range(runs_nb):

        for fold_i, (train_set, test_set) in enumerate(kfolds):

            if not folds_list is None and not fold_i in folds_list:
                continue

            # train context selector
            ner_model = pretrained_bert_for_token_classification(
                "bert-base-cased", train_set.tag_to_id
            )
            # HACK: split the training dataset in 2
            ctx_retrieval_ner_train_set, ctx_retrieval_gen_set = train_set.kfolds(
                2, shuffle=False
            )[0]
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
                        _run, ctx_retriever_model, "ctx_retriever_model"
                    )

            # PERFORMANCE HACK: only use the retrieval heuristic at
            # training time.
            train_set_heuristic = context_retriever_name_to_class[retrieval_heuristic](
                sents_nb=1
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
                    sacred_archive_huggingface_model(_run, ner_model, "ner_model")

            _run.log_scalar("gpu_usage", gpu_memory_usage())

            for inference_sents_nb in retrieval_heuristic_inference_sents_nb_list:

                neural_context_retriever = NeuralContextRetriever(
                    ctx_retriever_model,
                    retrieval_heuristic,
                    {"sents_nb": inference_sents_nb},
                    batch_size,
                    1,
                    use_cache=True,
                )
                ctx_test_set = neural_context_retriever(test_set)

                test_preds = predict(ner_model, ctx_test_set).tags
                precision, recall, f1 = score_ner(test_set.sents(), test_preds)
                _run.log_scalar(
                    f"run{run_i}.fold{fold_i}.test_precision",
                    precision,
                    step=inference_sents_nb,
                )
                _run.log_scalar(
                    f"run{run_i}.fold{fold_i}.test_recall",
                    recall,
                    step=inference_sents_nb,
                )
                _run.log_scalar(
                    f"run{run_i}.fold{fold_i}.test_f1", f1, step=inference_sents_nb
                )
