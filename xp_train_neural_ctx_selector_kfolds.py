import os
from typing import Any, Dict, List, Optional
from logging import Logger
from sacred.commands import print_config
from transformers import BertForTokenClassification  # type: ignore
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver, TelegramObserver
from conivel.train import train_ner_model
from conivel.datas.dataset import NERDataset
from conivel.datas.dekker import DekkerDataset
from conivel.datas.the_hunger_games import TheHungerGamesDataset
from conivel.datas.conll import CoNLLDataset
from conivel.datas.context import NeuralContextSelector
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


dataset_name_to_class = {
    "dekker": DekkerDataset,
    "the_hunger_games": TheHungerGamesDataset,
    "conll": CoNLLDataset,
}


@ex.config
def config():
    epochs_nb: int = 5
    learning_rate: float = 1e-5
    batch_size: int = 4
    max_examples_nb: Optional[int] = None
    examples_usefulness_threshold: float = 0.0
    heuristic_context_selector: str = "random"
    heuristic_context_selector_kwargs: Dict[str, Any]
    k: int
    train_datasets_names: List[str] = ["dekker"]


@ex.automain
def main(
    _run: Run,
    _log: Logger,
    epochs_nb: int,
    learning_rate: float,
    batch_size: int,
    max_examples_nb: Optional[int],
    examples_usefulness_threshold: float,
    heuristic_context_selector: str,
    heuristic_context_selector_kwargs: Dict[str, Any],
    k: int,
    train_datasets_names: List[str],
):
    print_config(_run)

    assert all([d in list(dataset_name_to_class.keys()) for d in train_datasets_names])

    dataset = NERDataset.concatenated(
        [dataset_name_to_class[name] for name in train_datasets_names]
    )
    kfolds = dataset.kfolds(k, shuffle=True, shuffle_seed=0)

    for i, (train_set, _) in enumerate(kfolds):

        model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=train_set.tags_nb,
            label2id=train_set.tag_to_id,
            id2label={v: k for k, v in train_set.tag_to_id.items()},
        )

        with RunLogScope(_run, f"fold{i}"):
            ner_model = train_ner_model(
                model,
                train_set,
                train_set,
                _run=_run,
                epochs_nb=2,
                batch_size=batch_size,
            )

            # generate context dataset
            ctx_dataset = NeuralContextSelector.generate_context_dataset(
                ner_model,
                train_set,
                batch_size,
                heuristic_context_selector,
                heuristic_context_selector_kwargs,
                max_examples_nb=max_examples_nb,
                examples_usefulness_threshold=examples_usefulness_threshold,
                _run=_run,
            )
            sacred_archive_jsonifiable_as_file(
                _run, ctx_dataset.to_jsonifiable(), "ctx_dataset"
            )

            # train neural context selector on generated context dataset
            ctx_selector = NeuralContextSelector.train_context_selector(
                ctx_dataset, epochs_nb, batch_size, learning_rate, _run=_run
            )
            sacred_archive_huggingface_model(_run, ctx_selector, "ctx_selector")  # type: ignore
