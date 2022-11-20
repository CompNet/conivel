import os, glob
from typing import Literal, Optional
from sacred import Experiment
from sacred.run import Run
from sacred.commands import print_config
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from conivel.datas.conll import CoNLLDataset
from conivel.datas.dataset import NERDataset
from conivel.datas.dekker import load_book
from conivel.train import train_ner_model
from conivel.utils import (
    pretrained_bert_for_token_classification,
    flattened,
    sacred_archive_jsonifiable_as_file,
)
from conivel.score import score_ner
from conivel.predict import predict
from conivel.analysis import get_errors


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

    # one of : "dekker", "conll"
    # - conll : load the usual CoNLL dataset
    # - dekker : load a Dekker dataset with PER/LOC/ORG annotation
    #           looks for files in `dataset_path` ending in
    #           ".conll.annotated"
    dataset_name: str = "dekker"

    # dataset root directory - used for dataset_name == "dekker" only
    dataset_path: Optional[str] = None

    # numebr of experiments repeat
    runs_nb: int = 5

    epochs_nb: int = 3
    batch_size: int


@ex.automain
def main(
    _run: Run,
    dataset_name: Literal["dekker", "conll"],
    dataset_path: Optional[str],
    runs_nb: int,
    epochs_nb: int,
    batch_size: int,
):
    print_config(_run)

    if dataset_name == "conll":
        train_dataset_full = CoNLLDataset("./conivel/datas/conll/train2.txt")
        train_dataset_per = CoNLLDataset(
            "./conivel/datas/conll/train2.txt", keep_only_classes={"PER"}
        )
        test_dataset = CoNLLDataset(
            "./conivel/datas/conll/test2.txt", keep_only_classes={"PER"}
        )
    elif dataset_name == "dekker":
        assert not dataset_path is None
        dataset_path = dataset_path.rstrip("/")
        # TODO: file names
        paths = glob.glob(f"{dataset_path}/*.conll.annotated")
        # TODO: hardcoded train/test split for now
        train_dataset_full = NERDataset([load_book(path) for path in paths[:8]])
        train_dataset_per = NERDataset(
            [load_book(path, keep_only_classes={"PER"}) for path in paths[:8]]
        )
        test_dataset = NERDataset(
            [load_book(path, keep_only_classes={"PER"}) for path in paths[8:]]
        )
    else:
        raise RuntimeError(f"unknown dataset : {dataset_name}")

    for run_i in range(runs_nb):

        for setup in ["full", "per"]:

            train_dataset = train_dataset_full if setup == "full" else train_dataset_per

            model = pretrained_bert_for_token_classification(
                "bert-base-cased", train_dataset.tag_to_id
            )
            model = train_ner_model(
                model,
                train_dataset,
                train_dataset,
                epochs_nb=epochs_nb,
                batch_size=batch_size,
            )

            preds = predict(model, test_dataset, batch_size=batch_size)
            precision, recall, f1 = score_ner(test_dataset.sents(), preds.tags)

            _run.log_scalar(f"{setup}_precision", precision)
            _run.log_scalar(f"{setup}_recall", recall)
            _run.log_scalar(f"{setup}_f1", f1)

            errors = flattened(
                [
                    get_errors(sent, ptags)
                    for sent, ptags in zip(test_dataset.sents(), preds.tags)
                ]
            )
            sacred_archive_jsonifiable_as_file(
                _run, [e.to_dict() for e in errors], f"{setup}_errors_{run_i}"
            )
