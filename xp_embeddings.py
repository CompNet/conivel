from typing import List, Optional, Tuple, Literal, Set
import os, glob
import torch
from sacred import Experiment
from sacred.run import Run
from sacred.commands import print_config
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from conivel.datas.dataset import NERDataset
from conivel.datas.conll import CoNLLDataset
from conivel.datas.dekker import load_book
from conivel.train import train_ner_model
from conivel.predict import predict
from conivel.utils import (
    pretrained_bert_for_token_classification,
    entities_from_bio_tags,
    sacred_archive_picklable_as_file,
)


script_dir = os.path.abspath(os.path.dirname(__file__))

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))
if os.path.isfile(f"{script_dir}/telegram_observer_config.json"):
    ex.observers.append(
        TelegramObserver.from_config(f"{script_dir}/telegram_observer_config.json")
    )


def load_dekker(
    dataset_path: str, keep_only_classes: Optional[Set[str]]
) -> Tuple[NERDataset, NERDataset]:
    """Load the version of Dekker dataset annotated with PER, LOC and ORG

    :param dataset_path: root directory
    :param keep_only_classes: passed to :func:`load_book`
    :return: ``(train set, test_set)``
    """
    dataset_path = dataset_path.rstrip("/")
    # TODO: file names
    paths = glob.glob(f"{dataset_path}/*.conll.annotated")
    # TODO: hardcoded train/test split for now
    train_dataset = NERDataset(
        [load_book(path, keep_only_classes=keep_only_classes) for path in paths[:11]]
    )
    test_dataset = NERDataset(
        [load_book(path, keep_only_classes=keep_only_classes) for path in paths[11:]]
    )
    return train_dataset, test_dataset


@ex.config
def config():
    # one of : "dekker", "conll"
    dataset_name: str
    dataset_path: str
    epochs_nb: int
    batch_size: int
    # Optional[List[str]]
    keep_only_classes: Optional[list] = None


@ex.automain
def main(
    _run: Run,
    dataset_name: Literal["dekker", "conll"],
    # path to Dekker et al dataset - use .annotated files for now
    # since ORG/LOC classes are important here
    dataset_path: str,
    epochs_nb: int,
    batch_size: int,
    keep_only_classes: Optional[List[str]],
):
    print_config(_run)

    koc = set(keep_only_classes) if not keep_only_classes is None else None

    if dataset_name == "dekker":
        train_dataset, test_dataset = load_dekker(dataset_path, koc)
    elif dataset_name == "conll":
        train_dataset = CoNLLDataset.train_dataset(keep_only_classes=keep_only_classes)
        test_dataset = CoNLLDataset.test_dataset(keep_only_classes=keep_only_classes)
    else:
        raise RuntimeError(f"Unknown dataset {dataset_name}")

    model = pretrained_bert_for_token_classification(
        "bert-base-cased", train_dataset.tag_to_id
    )
    model = train_ner_model(
        model,
        train_dataset,
        train_dataset,
        _run,
        epochs_nb=epochs_nb,
        batch_size=batch_size,
    )

    preds = predict(
        model,
        test_dataset,
        batch_size=batch_size,
        additional_outputs={"embeddings"},
        transfer_additional_outputs_to_cpu=True,
    )
    assert not preds.embeddings is None

    entities_embeddings = {}
    for sent, sent_embeddings in zip(test_dataset.sents(), preds.embeddings):
        sent_entities = entities_from_bio_tags(sent.tokens, sent.tags)
        for entity in sent_entities:
            entities_embeddings[entity] = torch.mean(
                sent_embeddings[entity.start_idx : entity.end_idx + 1], dim=0
            )

    sacred_archive_picklable_as_file(_run, entities_embeddings, "entities_embeddings")
