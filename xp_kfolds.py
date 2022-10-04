import os
from typing import Dict, List, Optional, Union
import shutil
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from transformers import BertForTokenClassification  # type: ignore
from conivel.datas.dekker import DekkerDataset
from conivel.datas.context import context_selector_name_to_class
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
    k: int = 5
    shuffle_kfolds_seed: Optional[int] = None
    save_models: bool = True
    book_group: Optional[str] = None


@ex.automain
def main(
    _run: Run,
    context_selectors: Union[Dict[str, dict], List[Dict[str, dict]]],
    epochs_nb: int,
    k: int,
    shuffle_kfolds_seed: Optional[int],
    save_models: bool,
    book_group: Optional[str],
):
    print_config(_run)

    dekker_dataset = DekkerDataset(book_group=book_group)
    kfolds = dekker_dataset.kfolds(
        k, shuffle=not shuffle_kfolds_seed is None, shuffle_seed=shuffle_kfolds_seed
    )

    selectors = [
        context_selector_name_to_class[key](**value)
        for key, value in context_selectors.items()
    ]
    for train_set, test_set in kfolds:
        train_set.context_selectors = selectors
        test_set.context_selectors = selectors

    for i, (train_set, test_set) in enumerate(kfolds):

        # train
        with RunLogScope(_run, f"fold{i}"):

            model = BertForTokenClassification.from_pretrained(
                "bert-base-cased",
                num_labels=train_set.tags_nb,
                label2id=train_set.tag_to_id,
                id2label={v: k for k, v in train_set.tag_to_id.items()},
            )

            model = train_ner_model(
                model,
                train_set,
                test_set,
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

        # test
        test_preds = predict(model, test_set).tags
        precision, recall, f1 = score_ner(
            test_set.sents(), test_preds, ignored_classes={"MISC", "ORG", "LOC"}
        )
        _run.log_scalar("test_precision", precision)
        _run.log_scalar("test_recall", recall)
        _run.log_scalar("test_f1", f1)
