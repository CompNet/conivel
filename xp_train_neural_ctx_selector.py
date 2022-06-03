from typing import Optional
from sacred.commands import print_config
from transformers import BertForTokenClassification  # type: ignore
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver
from conivel.datas.dataset import NERDataset
from conivel.datas.dekker import DekkerDataset
from conivel.datas.conll import CoNLLDataset
from conivel.datas.context import NeuralContextSelector
from conivel.predict import predict
from conivel.utils import (
    sacred_archive_huggingface_model,
    sacred_archive_jsonifiable_as_file,
)


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))


dataset_to_constructor = {"conll": CoNLLDataset.train_dataset, "dekker": DekkerDataset}


@ex.config
def config():
    ner_model_path: str
    ner_train_dataset_name: str = "conll"
    epochs_nb: int = 5
    learning_rate: float = 1e-5
    batch_size: int = 4
    samples_per_sent: int = 4
    max_examples_nb: Optional[int] = None


@ex.automain
def main(
    _run: Run,
    ner_model_path: str,
    ner_train_dataset_name: str,
    epochs_nb: int,
    learning_rate: float,
    batch_size: int,
    samples_per_sent: int,
    max_examples_nb: Optional[int],
):
    print_config(_run)
    assert ner_train_dataset_name in dataset_to_constructor.keys()

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
        samples_per_sent,
        max_examples_nb=max_examples_nb,
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
