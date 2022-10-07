import os, gc, copy
from typing import Dict, List, Optional, Union
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from transformers import BertForTokenClassification  # type: ignore
from conivel.datas.dekker import DekkerDataset
from conivel.datas.context import NeuralContextSelector, context_selector_name_to_class
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import (
    RunLogScope,
    sacred_archive_huggingface_model,
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
    # list of folds number (starting from 0) to perform the experiment
    # on. If not specified, perform the experiment on all folds
    folds_list: Optional[list] = None

    # -- common parameters
    batch_size: int
    # wether models should be saved or not
    save_models: bool = True
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
    # usefulness threshold for context retrieval examples (examples
    # with abs(usefulness) lower than this value are discarded)
    ctx_retrieval_usefulness_threshold: float = 0.1

    # -- NER trainng parameters
    # min number of context sents
    min_sents_nb: int = 1
    # max number of context sents
    max_sents_nb: int = 8
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
    retrieval_heuristic: str,
    retrieval_heuristic_gen_kwargs: dict,
    retrieval_heuristic_inference_kwargs: dict,
    ctx_retrieval_epochs_nb: int,
    ctx_retrieval_lr: float,
    ctx_retrieval_usefulness_threshold: float,
    min_sents_nb: int,
    max_sents_nb: int,
    ner_epochs_nb: int,
    ner_lr: float,
):
    assert retrieval_heuristic in ["random", "bm25", "sameword"]
    print_config(_run)

    dekker_dataset = DekkerDataset(book_group=book_group)
    kfolds = dekker_dataset.kfolds(
        k, shuffle=not shuffle_kfolds_seed is None, shuffle_seed=shuffle_kfolds_seed
    )

    for fold_i, (train_set, test_set) in enumerate(kfolds):

        if not folds_list is None and not fold_i in folds_list:
            continue

        # train context selector
        ner_model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=train_set.tags_nb,
            label2id=train_set.tag_to_id,
            id2label={v: k for k, v in train_set.tag_to_id.items()},
        )

        # HACK: split the training dataset in 2
        ctx_retrieval_ner_train_set, ctx_retrieval_gen_set = train_set.kfolds(
            2, shuffle=False
        )[0]

        with RunLogScope(_run, f"ctx_retrieval_training.fold{fold_i}"):

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
            ctx_retrieval_dataset = NeuralContextSelector.generate_context_dataset(
                ner_model,
                ctx_retrieval_gen_set,
                batch_size,
                retrieval_heuristic,
                retrieval_heuristic_gen_kwargs,
                examples_usefulness_threshold=ctx_retrieval_usefulness_threshold,
                _run=_run,
            )
            sacred_archive_jsonifiable_as_file(
                _run, ctx_retrieval_dataset.to_jsonifiable(), "ctx_retrieval_dataset"
            )

            # NER model is not needed anymore : free the GPU of it
            del ner_model
            gc.collect()

            # train a context retriever using the previously generated
            # context retrieval dataset
            ctx_retriever_model = NeuralContextSelector.train_context_selector(
                ctx_retrieval_dataset,
                ctx_retrieval_epochs_nb,
                batch_size,
                ctx_retrieval_lr,
                _run,
            )
            if save_models:
                sacred_archive_huggingface_model(
                    _run, ctx_retriever_model, "ctx_retriever_model"
                )

        # PERFORMANCE HACK: only use the retrieval heuristic at
        # training time. At training time, the number of sentences
        # retrieved is random between ``min_sents_nb`` and
        # ``max_sents_nb`` for each example.
        train_set_heuristic_kwargs = copy.deepcopy(retrieval_heuristic_inference_kwargs)
        train_set_heuristic_kwargs["sents_nb"] = list(
            range(min_sents_nb, max_sents_nb + 1)
        )
        train_set_heuristic = context_selector_name_to_class[retrieval_heuristic](
            **train_set_heuristic_kwargs
        )
        train_set.context_selectors = [train_set_heuristic]

        # train ner model on train_set
        ner_model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=train_set.tags_nb,
            label2id=train_set.tag_to_id,
            id2label={v: k for k, v in train_set.tag_to_id.items()},
        )
        with RunLogScope(_run, f"ner.fold{fold_i}"):
            ner_model = train_ner_model(
                ner_model,
                train_set,
                train_set,
                _run=_run,
                epochs_nb=ner_epochs_nb,
                batch_size=batch_size,
                learning_rate=ner_lr,
            )
            if save_models:
                sacred_archive_huggingface_model(_run, ner_model, "ner_model")

        for sents_nb in range(min_sents_nb, max_sents_nb + 1):

            neural_context_retriever = NeuralContextSelector(
                ctx_retriever_model,
                retrieval_heuristic,
                retrieval_heuristic_inference_kwargs,
                batch_size,
                sents_nb,
            )
            test_set.context_selectors = [neural_context_retriever]

            test_preds = predict(ner_model, test_set).tags
            precision, recall, f1 = score_ner(test_set.sents(), test_preds)
            _run.log_scalar(f"test_precision.fold{fold_i}", precision, step=sents_nb)
            _run.log_scalar(f"test_recall.fold{fold_i}", recall, step=sents_nb)
            _run.log_scalar(f"test_f1.fold{fold_i}", f1, step=sents_nb)
