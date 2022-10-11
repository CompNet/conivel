import os
from sacred import Experiment
from sacred.run import Run
from sacred.commands import print_config
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
from transformers import BertForTokenClassification  # type: ignore
from conivel.datas.dekker.dekker import DekkerDataset
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import RunLogScope, gpu_memory_usage


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
    repeats_nb: int = 5
    low_epochs_nb: int
    high_epochs_nb: int
    batch_size: int


@ex.automain
def main(
    _run: Run, repeats_nb: int, low_epochs_nb: int, high_epochs_nb: int, batch_size: int
):
    print_config(_run)

    train_set, test_set = DekkerDataset().kfolds(5, False)[0]

    for repeat_i in range(repeats_nb):

        with RunLogScope(_run, f"low_run{repeat_i}"):

            ner_model = BertForTokenClassification.from_pretrained(
                "bert-base-cased",
                num_labels=train_set.tags_nb,
                label2id=train_set.tag_to_id,
                id2label={v: k for k, v in train_set.tag_to_id.items()},
            )

            ner_model = train_ner_model(
                ner_model,
                train_set,
                train_set,
                _run,
                low_epochs_nb,
                batch_size=batch_size,
                learning_rate=2e-5,
            )

        test_preds = predict(ner_model, test_set).tags
        precision, recall, f1 = score_ner(test_set.sents(), test_preds)
        _run.log_scalar(f"low_test_precision", precision)
        _run.log_scalar(f"low_test_recall", recall)
        _run.log_scalar(f"low_test_f1", f1)

        _run.log_scalar("gpu_usage", gpu_memory_usage())

    for repeat_i in range(repeats_nb):

        with RunLogScope(_run, f"high_run{repeat_i}"):

            ner_model = BertForTokenClassification.from_pretrained(
                "bert-base-cased",
                num_labels=train_set.tags_nb,
                label2id=train_set.tag_to_id,
                id2label={v: k for k, v in train_set.tag_to_id.items()},
            )

            ner_model = train_ner_model(
                ner_model,
                train_set,
                train_set,
                _run,
                high_epochs_nb,
                batch_size=batch_size,
                learning_rate=2e-5,
            )

        test_preds = predict(ner_model, test_set).tags
        precision, recall, f1 = score_ner(test_set.sents(), test_preds)
        _run.log_scalar(f"high_test_precision", precision)
        _run.log_scalar(f"high_test_recall", recall)
        _run.log_scalar(f"high_test_f1", f1)

        _run.log_scalar("gpu_usage", gpu_memory_usage())
