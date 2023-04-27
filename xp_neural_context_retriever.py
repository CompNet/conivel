import os, random, sys
from typing import List, Literal, Optional, Tuple
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from sacred.observers import FileStorageObserver, TelegramObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from nltk.tokenize import word_tokenize
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset
from conivel.datas.dekker import DekkerDataset
from conivel.datas.context import (
    ContextRetrievalDataset,
    ContextRetrievalExample,
    NeuralContextRetriever,
)
from conivel.predict import predict
from conivel.score import score_ner
from conivel.train import train_ner_model
from conivel.utils import (
    NEREntity,
    entities_from_bio_tags,
    RunLogScope,
    sacred_archive_huggingface_model,
    sacred_archive_jsonifiable_as_file,
    gpu_memory_usage,
    pretrained_bert_for_token_classification,
    flattened,
)

script_dir = os.path.abspath(os.path.dirname(__file__))

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))
if os.path.isfile(f"{script_dir}/telegram_observer_config.json"):
    ex.observers.append(
        TelegramObserver.from_config(f"{script_dir}/telegram_observer_config.json")
    )


def request_alpaca(
    alpaca, tokenizer, prompt: str, device_str: Literal["cpu", "cuda"]
) -> List[str]:
    device = torch.device(device_str)

    prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    t_prompt = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = alpaca.generate(
        t_prompt, max_new_tokens=200, do_sample=True, top_k=50, top_p=0.95
    )

    out_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    try:
        marker = "### Response:\n"
        out_text = out_text[out_text.index(marker) + len(marker) :]
    except ValueError as e:
        print(e)

    return word_tokenize(out_text)


def generate_pos_example(
    alpaca,
    tokenizer,
    sent: NERSentence,
    entity: NEREntity,
    device: Literal["cpu", "cuda"],
) -> ContextRetrievalExample:

    sent_text = " ".join(sent.tokens)
    entity_text = " ".join(entity.tokens)

    PROMPTS = {
        "PER": [
            f"'{sent_text}' - In the preceding sentence, {entity_text} is a character. Invent a one-sentence description for this character, mentioning their name.",
            f"'{sent_text}' - In the preceding sentence, {entity_text} is a character. Invent a single sentence depicting this character performing an action, mentioning their name.",
        ],
        "LOC": [
            f"'{sent_text}' - In the preceding sentence, {entity_text} is a location. Invent a one-sentence description for this location, mentioning its name.",
            f"Invent a single sentence depicting a character going to {entity_text}.",
        ],
        "ORG": [
            f"'{sent_text}' - In the preceding sentence, {entity_text} is an organisation. Invent a one-sentence description for this organisation, mentioning its name."
        ],
    }

    example_text = request_alpaca(
        alpaca, tokenizer, random.choice(PROMPTS[entity.tag]), device
    )
    return ContextRetrievalExample(sent.tokens, sent.tags, example_text, [], "right", 1)


def generate_pos_examples(
    dataset: NERDataset, alpaca, tokenizer, device: Literal["cpu", "cuda"]
) -> List[ContextRetrievalExample]:

    # { entity_str => ex }
    exs = {}

    print("starting alpaca.cpp...")

    t = tqdm(dataset.sents())

    for sent in t:

        for entity in entities_from_bio_tags(sent.tokens, sent.tags):

            entity_str = " ".join(entity.tokens)

            if not entity_str in exs:
                exs[entity_str] = generate_pos_example(
                    alpaca, tokenizer, sent, entity, device
                )

            t.set_description(f"{len(exs)} examples")

    return list(exs.values())


def generate_neg_examples_negsampling(
    dataset: NERDataset,
) -> List[ContextRetrievalExample]:

    # { entity_str => ex }
    exs = {}

    t = tqdm(enumerate(dataset.documents))
    for doc_i, doc in t:

        other_doc_sents = list(
            flattened([d for i, d in enumerate(dataset.documents) if i != doc_i])
        )

        for sent in doc:

            for entity in entities_from_bio_tags(sent.tokens, sent.tags):

                entity_str = " ".join(entity.tokens)
                if entity_str in exs:
                    continue

                other_sent = random.choice(other_doc_sents)
                exs[entity_str] = ContextRetrievalExample(
                    sent.tokens,
                    sent.tags,
                    other_sent.tokens,
                    other_sent.tags,
                    "right",
                    0,
                )
                t.set_description(f"{len(exs)} examples")

    return list(exs.values())


def generate_neg_examples_othercontexts(
    examples: List[ContextRetrievalExample],
) -> List[ContextRetrievalExample]:
    neg_examples = []

    for ex in examples:

        forbidden_entities = [
            " ".join(e.tokens).lower()
            for e in entities_from_bio_tags(ex.sent, ex.sent_tags)
        ]

        other_examples: List[ContextRetrievalExample] = []
        for other_ex in examples:
            other_ex_entities = entities_from_bio_tags(
                other_ex.sent, other_ex.sent_tags
            )
            other_ex_entities = [
                " ".join(ex_entity.tokens).lower() for ex_entity in other_ex_entities
            ]
            if any(
                [
                    ex_entity in forbidden_ent or forbidden_ent in ex_entity
                    for ex_entity in other_ex_entities
                    for forbidden_ent in forbidden_entities
                ]
            ):
                continue
            other_examples.append(other_ex)

        oex = random.choice(other_examples)

        neg_examples.append(
            ContextRetrievalExample(
                ex.sent, ex.sent_tags, oex.context, oex.sent_tags, "right", 0
            )
        )

    return neg_examples


def gen_cr_dataset(
    dataset: NERDataset, alpaca_model_str: str, device_str: Literal["cpu", "cuda"]
) -> ContextRetrievalDataset:
    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(alpaca_model_str)
    alpaca = AutoModelForCausalLM.from_pretrained(alpaca_model_str).to(device)

    # n
    pos_exs = generate_pos_examples(dataset, alpaca, tokenizer, device_str)
    # n
    neg_exs = generate_neg_examples_negsampling(dataset)
    # 2n
    neg_exs += generate_neg_examples_othercontexts(pos_exs)
    # balance dataset
    neg_exs = random.choices(neg_exs, k=len(pos_exs))

    return ContextRetrievalDataset(pos_exs + neg_exs)


def gen_cr_dataset_kfolds(
    _run: Run,
    ner_kfolds: List[Tuple[NERDataset, NERDataset]],
    alpaca_model_str: str,
    device: Literal["cpu", "cuda"],
) -> List[Tuple[ContextRetrievalDataset, ContextRetrievalDataset]]:

    test_cr_datasets = []
    for fold_i, (_, test) in enumerate(ner_kfolds):
        cr_dataset = gen_cr_dataset(test, alpaca_model_str, device)
        test_cr_datasets.append(cr_dataset)
        sacred_archive_jsonifiable_as_file(
            _run, cr_dataset.to_jsonifiable(), f"fold{fold_i}.cr_test_dataset"
        )

    # for each test dataset, the train dataset for this fold is the
    # concatenation of all other folds tests datasets
    train_cr_datasets = [
        ContextRetrievalDataset.concatenated([t for t in test_cr_datasets if t != test])
        for test in test_cr_datasets
    ]
    for fold_i, train in enumerate(train_cr_datasets):
        sacred_archive_jsonifiable_as_file(
            _run, train.to_jsonifiable(), f"fold{fold_i}.cr_train_dataset"
        )

    return [(train, test) for train, test in zip(train_cr_datasets, test_cr_datasets)]


@ex.config
def config():
    # -- datas parameters
    # number of folds
    k: int = 5
    # seed to use when shuffling folds. If ``None``, no shuffling is
    # performed.
    shuffle_kfolds_seed: Optional[int] = None

    # -- common parameters
    batch_size: int
    # wether models should be saved or not
    save_models: bool = True
    # number of experiment repeats
    runs_nb: int

    # -- context retrieval parameters
    # alpaca model used for generation
    # 'chavinlo/alpaca-native'
    # 'chavinlo/gpt4-x-alpaca'
    cr_gen_alpaca_model: str = "chavinlo/gpt4-x-alpaca"
    # device to use when generating examples - either 'cpu' or 'cuda'
    cr_gen_device: str = "cpu"
    # number of epochs for context retrieval training
    cr_epochs_nb: int = 3
    # learning rate for context retrieval training
    cr_lr: float = 2e-5
    # dropout for context retrieval training
    cr_dropout: float = 0.1

    # -- NER parameters
    ner_epochs_nb: int = 2
    ner_lr: float = 2e-5


@ex.automain
def main(
    _run: Run,
    k: int,
    shuffle_kfolds_seed: Optional[int],
    batch_size: int,
    save_models: bool,
    runs_nb: int,
    cr_gen_alpaca_model: str,
    cr_gen_device: Literal["cpu", "cuda"],
    cr_epochs_nb: int,
    cr_lr: float,
    cr_dropout: float,
    ner_epochs_nb: int,
    ner_lr: float,
):
    print_config(_run)

    dataset = DekkerDataset()
    ner_kfolds = dataset.kfolds(
        k, shuffle=not shuffle_kfolds_seed is None, shuffle_seed=shuffle_kfolds_seed
    )
    folds_nb = len(ner_kfolds)

    cr_kfolds = gen_cr_dataset_kfolds(
        _run, ner_kfolds, cr_gen_alpaca_model, cr_gen_device
    )

    # * Metrics matrices
    #   each matrix is of shape (runs_nb, folds_nb, sents_nb)
    #   these are used to record mean metrics across folds, runs...
    cr_precision_matrix = np.zeros((runs_nb, folds_nb))
    cr_recall_matrix = np.zeros((runs_nb, folds_nb))
    cr_f1_matrix = np.zeros((runs_nb, folds_nb))
    ner_precision_matrix = np.zeros((runs_nb, folds_nb))
    ner_recall_matrix = np.zeros((runs_nb, folds_nb))
    ner_f1_matrix = np.zeros((runs_nb, folds_nb))
    metrics_matrices: List[Tuple[str, np.ndarray]] = [
        ("cr_precision", cr_precision_matrix),
        ("cr_recall", cr_recall_matrix),
        ("cr_f1", cr_f1_matrix),
        ("ner_precision", ner_precision_matrix),
        ("ner_recall", ner_recall_matrix),
        ("ner_f1", ner_f1_matrix),
    ]

    for run_i in range(runs_nb):

        for fold_i, ((ner_train, ner_test), (cr_train, cr_test)) in enumerate(
            zip(ner_kfolds, cr_kfolds)
        ):

            _run.log_scalar("gpu_usage", gpu_memory_usage())

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.cr_model_training"):

                neural_retriever_model = NeuralContextRetriever.train_context_selector(
                    cr_train,
                    cr_epochs_nb,
                    batch_size,
                    cr_lr,
                    _run=_run,
                    log_full_loss=True,
                    dropout=cr_dropout,
                )
                neural_retriever = NeuralContextRetriever(
                    neural_retriever_model,
                    "all",
                    {"sents_nb": 1},  # WARNING: ignored
                    batch_size,
                    1,
                )
                if save_models:
                    sacred_archive_huggingface_model(
                        _run, neural_retriever_model, "cr_model"  # type: ignore
                    )

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.cr_model_testing"):

                # (len(test_ctx_retrieval), 2)
                raw_preds = neural_retriever.predict(cr_test)
                preds = torch.argmax(raw_preds, dim=1).cpu()
                labels = cr_test.labels()
                assert not labels is None

                # * Micro F1
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, preds, average="micro"
                )

                cr_precision_matrix[run_i][fold_i] = precision
                cr_recall_matrix[run_i][fold_i] = recall
                cr_f1_matrix[run_i][fold_i] = f1
                _run.log_scalar(f"precision", precision)
                _run.log_scalar(f"recall", recall)
                _run.log_scalar(f"f1", f1)

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.ner_model_training"):

                train_and_ctx = neural_retriever(ner_train, quiet=False)
                ner_model = pretrained_bert_for_token_classification(
                    "bert-base-cased", ner_train.tag_to_id
                )
                ner_model = train_ner_model(
                    ner_model,
                    train_and_ctx,
                    train_and_ctx,
                    _run,
                    epochs_nb=ner_epochs_nb,
                    batch_size=batch_size,
                    learning_rate=ner_lr,
                )

                if save_models:
                    sacred_archive_huggingface_model(_run, ner_model, f"ner_model")

            with RunLogScope(_run, f"run{run_i}.fold{fold_i}.ner_model_testing"):

                test_and_ctx = neural_retriever(ner_test, quiet=False)
                preds = predict(ner_model, test_and_ctx, batch_size=batch_size).tags
                precision, recall, f1 = score_ner(ner_test.sents(), preds)

                ner_precision_matrix[run_i][fold_i] = precision
                ner_recall_matrix[run_i][fold_i] = recall
                ner_f1_matrix[run_i][fold_i] = f1
                _run.log_scalar("precision", precision)
                _run.log_scalar(f"recall", recall)
                _run.log_scalar(f"f1", f1)

        # * Run mean metrics
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                _run.log_scalar(
                    f"run{run_i}.{op_name}_test_{metrics_name}",
                    op(matrix[run_i], axis=0),
                )

    # * Folds mean metrics
    for fold_i in range(folds_nb):
        for metrics_name, matrix in metrics_matrices:
            for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
                _run.log_scalar(
                    f"fold{fold_i}.{op_name}_test_{metrics_name}",
                    op(matrix[:, fold_i], axis=0),
                )

    # * Global mean metrics
    for name, matrix in metrics_matrices:
        for op_name, op in [("mean", np.mean), ("stdev", np.std)]:
            _run.log_scalar(
                f"{op_name}_test_{name}",
                op(matrix, axis=(0, 1)),
            )
