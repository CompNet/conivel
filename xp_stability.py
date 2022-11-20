from collections import defaultdict
import os, glob
from typing import List, Optional, Set, Dict
from sacred import Experiment
from sacred.run import Run
from sacred.commands import print_config
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
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


@ex.config
def config():
    dataset_path: str
    epochs_nb: int
    batch_size: int
    runs_nb: int
    # Optional[List[str]]
    keep_only_classes: Optional[list] = None


@ex.automain
def main(
    _run: Run,
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

    for run_i in range(runs_nb):

        dataset_path = dataset_path.rstrip("/")
        # TODO: file names
        paths = glob.glob(f"{dataset_path}/*.conll.annotated")
        # TODO: hardcoded train/test split for now
        koc = set(keep_only_classes) if not keep_only_classes is None else None
        train_dataset = NERDataset(
            [load_book(path, keep_only_classes=koc) for path in paths[:8]]
        )
        test_dataset = NERDataset(
            [load_book(path, keep_only_classes=koc) for path in paths[8:]]
        )

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
        _run.log_scalar("recall", recall)
        _run.log_scalar("f1", f1)

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
