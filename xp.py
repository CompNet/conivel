import os
from typing import Dict, Optional
import shutil
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from transformers import BertForTokenClassification  # type: ignore
from conivel.datas.conll import CoNLLDataset
from conivel.datas.dekker import DekkerDataset
from conivel.datas.context import context_retriever_name_to_class
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import RunLogScope


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
    context_selectors: Dict[str, dict] = {}
    epochs_nb: int = 2
    repeats_nb: int = 1
    save_models: bool = True
    book_group: Optional[str] = None


@ex.automain
def main(
    _run: Run,
    context_selectors: Dict[str, dict],
    epochs_nb: int,
    repeats_nb: int,
    save_models: bool,
    book_group: Optional[str],
):
    print(context_selectors)
    print_config(_run)

    selectors = [
        context_retriever_name_to_class[key](**value)
        for key, value in context_selectors.items()
    ]

    train_dataset = CoNLLDataset.train_dataset(context_selectors=selectors)
    valid_dataset = CoNLLDataset.valid_dataset(context_selectors=selectors)
    test_dataset = CoNLLDataset.test_dataset(context_selectors=selectors)
    dekker_dataset = DekkerDataset(context_selectors=selectors, book_group=book_group)

    for i in range(repeats_nb):

        with RunLogScope(_run, f"train{i}"):

            model = BertForTokenClassification.from_pretrained(
                "bert-base-cased",
                num_labels=train_dataset.tags_nb,
                label2id=train_dataset.tag_to_id,
                id2label={v: k for k, v in train_dataset.tag_to_id.items()},
            )

            model = train_ner_model(
                model,
                train_dataset,
                valid_dataset,
                _run=_run,
                epochs_nb=epochs_nb,
                ignored_valid_classes={"MISC", "ORG", "LOC"},
            )
            if save_models:
                model.save_pretrained("./model")
                shutil.make_archive("./model", "gztar", ".", "model")
                _run.add_artifact("./model.tar.gz")
                shutil.rmtree("./model")
                os.remove("./model.tar.gz")

        # CoNLL test
        test_preds = predict(model, test_dataset).tags
        precision, recall, f1 = score_ner(
            test_dataset.sents(), test_preds, ignored_classes={"MISC", "ORG", "LOC"}
        )
        _run.log_scalar("test_precision", precision)
        _run.log_scalar("test_recall", recall)
        _run.log_scalar("test_f1", f1)

        # Dekker test
        dekker_preds = predict(model, dekker_dataset).tags
        precision, recall, f1 = score_ner(
            dekker_dataset.sents(),
            dekker_preds,
            ignored_classes={"MISC", "ORG", "LOC"},
        )
        _run.log_scalar("dekker_precision", precision)
        _run.log_scalar("dekker_recall", recall)
        _run.log_scalar("dekker_f1", f1)
