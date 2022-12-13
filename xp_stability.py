from collections import defaultdict
import os, glob
from typing import List, Literal, Optional, Set, Dict, Tuple
from conivel.datas.conll.conll import CoNLLDataset
from sacred import Experiment
from sacred.run import Run
from sacred.commands import print_config
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
import numpy as np
from conivel.datas.dataset import NERDataset
from conivel.datas.dekker import load_book
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import (
    NEREntity,
    entities_from_bio_tags,
    flattened,
    pretrained_bert_for_token_classification,
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
    runs_nb: int
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
    runs_nb: int,
    keep_only_classes: Optional[List[str]],
):
    print_config(_run)

    entity_to_recall: Dict[NEREntity, List[bool]] = defaultdict(list)

    precision_matrix = np.zeros((runs_nb,))
    recall_matrix = np.zeros((runs_nb,))
    f1_matrix = np.zeros((runs_nb,))

    for run_i in range(runs_nb):

        koc = set(keep_only_classes) if not keep_only_classes is None else None

        if dataset_name == "dekker":
            train_dataset, test_dataset = load_dekker(dataset_path, koc)
        elif dataset_name == "conll":
            train_dataset = CoNLLDataset.train_dataset(
                keep_only_classes=keep_only_classes
            )
            test_dataset = CoNLLDataset.test_dataset(
                keep_only_classes=keep_only_classes
            )
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

        preds = predict(model, test_dataset, batch_size=batch_size)
        precision, recall, f1 = score_ner(test_dataset.sents(), preds.tags)
        _run.log_scalar("precision", precision)
        precision_matrix[run_i] = precision
        _run.log_scalar("recall", recall)
        recall_matrix[run_i] = recall
        _run.log_scalar("f1", f1)
        f1_matrix[run_i] = f1

        tokens = flattened([sent.tokens for sent in test_dataset.sents()])
        true_tags = flattened([sent.tags for sent in test_dataset.sents()])
        true_entities = entities_from_bio_tags(tokens, true_tags)
        pred_entities = entities_from_bio_tags(tokens, flattened(preds.tags))

        for true_ent in true_entities:
            entity_to_recall[true_ent].append(true_ent in pred_entities)

    def stability(entity_was_recalled: List[bool]) -> float:
        recalls_nb = len([e for e in entity_was_recalled if e])
        no_recalls_nb = len(entity_was_recalled) - recalls_nb
        return abs(recalls_nb - no_recalls_nb) / len(entity_was_recalled)

    entity_to_stability = [
        {"entity": vars(ent), "stability": stability(was_recalled)}
        for ent, was_recalled in entity_to_recall.items()
    ]
    sacred_archive_jsonifiable_as_file(_run, entity_to_stability, "entity_to_stability")

    _run.log_scalar("mean_precision", np.mean(precision_matrix))
    _run.log_scalar("mean_recall", np.mean(recall_matrix))
    _run.log_scalar("mean_f1", np.mean(f1_matrix))
