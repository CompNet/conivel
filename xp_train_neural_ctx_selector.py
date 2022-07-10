import os
from typing import Any, Dict, Optional
from logging import Logger
from sacred.commands import print_config
from transformers import BertForTokenClassification  # type: ignore
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver, TelegramObserver
from conivel.datas.dataset import NERDataset
from conivel.datas.dekker import DekkerDataset
from conivel.datas.conll import CoNLLDataset
from conivel.datas.context import NeuralContextSelector
from conivel.utils import (
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


dataset_to_constructor = {"conll": CoNLLDataset.train_dataset, "dekker": DekkerDataset}


@ex.config
def config():
    ner_model_path: str
    ner_train_dataset_name: str = "conll"
    epochs_nb: int = 5
    learning_rate: float = 1e-5
    batch_size: int = 4
    max_examples_nb: Optional[int] = None
    examples_usefulness_threshold: float = 0.0
    heuristic_context_selector: str
    heuristic_context_selector_kwargs: Dict[str, Any]


@ex.automain
def main(
    _run: Run,
    _log: Logger,
    ner_model_path: str,
    ner_train_dataset_name: str,
    epochs_nb: int,
    learning_rate: float,
    batch_size: int,
    max_examples_nb: Optional[int],
    examples_usefulness_threshold: float,
    heuristic_context_selector: str,
    heuristic_context_selector_kwargs: Dict[str, Any],
):
    print_config(_run)

    assert ner_train_dataset_name in dataset_to_constructor.keys()
    if ner_train_dataset_name == "dekker":
        _log.warning(
            "you are trying to use the 'dekker' NER dataset to train a context selector. This has not been tested and may fail due to its usage of only PER tags."
        )

    ner_train_dataset: NERDataset = dataset_to_constructor[ner_train_dataset_name]()
    ner_model = BertForTokenClassification.from_pretrained(
        ner_model_path,
        num_labels=ner_train_dataset.tags_nb,
        label2id=ner_train_dataset.tag_to_id,
        id2label={v: k for k, v in ner_train_dataset.tag_to_id.items()},
    )

    # generate context dataset
    ctx_dataset = NeuralContextSelector.generate_context_dataset(
        ner_model,
        ner_train_dataset,
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
