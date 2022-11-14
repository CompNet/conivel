from typing import List, Optional
import os
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from conivel.datas.dekker import DekkerDataset
from conivel.datas.context import context_retriever_name_to_class
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import (
    RunLogScope,
    gpu_memory_usage,
    sacred_archive_huggingface_model,
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
    context_retriever_kwargs: dict

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
    k: int,
    shuffle_kfolds_seed: Optional[int],
    book_group: Optional[str],
    batch_size: int,
    save_models: bool,
    runs_nb: int,
    context_retriever: str,
    context_retriever_kwargs: dict,
    sents_nb_list: List[int],
    ner_epochs_nb: int,
    ner_lr: float,
):
    print_config(_run)

    dekker_dataset = DekkerDataset(book_group=book_group)
    kfolds = dekker_dataset.kfolds(
        k, shuffle=not shuffle_kfolds_seed is None, shuffle_seed=shuffle_kfolds_seed
    )

    for run_i in range(runs_nb):

        for fold_i, (train_set, test_set) in enumerate(kfolds):

            ctx_retriever = context_retriever_name_to_class[context_retriever](
                sents_nb=sents_nb_list, **context_retriever_kwargs
            )
            ctx_train_set = ctx_retriever(train_set)

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
                )
                if save_models:
                    sacred_archive_huggingface_model(_run, model, "model")  # type: ignore

            for sents_nb in sents_nb_list:

                _run.log_scalar("gpu_usage", gpu_memory_usage())

                ctx_retriever = context_retriever_name_to_class[context_retriever](
                    sents_nb=sents_nb, **context_retriever_kwargs
                )
                ctx_test_set = ctx_retriever(test_set)

                # test
                test_preds = predict(model, ctx_test_set, batch_size=batch_size).tags
                precision, recall, f1 = score_ner(ctx_test_set.sents(), test_preds)
                _run.log_scalar(
                    f"run{run_i}.fold{fold_i}.test_precision",
                    precision,
                    step=sents_nb,
                )
                _run.log_scalar(
                    f"run{run_i}.fold{fold_i}.test_recall", recall, step=sents_nb
                )
                _run.log_scalar(f"run{run_i}.fold{fold_i}.test_f1", f1, step=sents_nb)
