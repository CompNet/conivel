from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Set, Type, Union, cast, Tuple
import random, copy
from dataclasses import dataclass
import nltk
from sacred.run import Run
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForTokenClassification, BertForSequenceClassification, BertTokenizerFast, DataCollatorWithPadding  # type: ignore
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from rank_bm25 import BM25Okapi
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset
from conivel.utils import (
    NEREntity,
    entities_from_bio_tags,
    get_tokenizer,
    sent_with_ctx_from_matchs,
)
from conivel.predict import predict


@dataclass
class ContextRetrievalMatch:
    sentence: NERSentence
    sentence_idx: int
    side: Literal["left", "right"]
    score: Optional[float]


class ContextRetriever:
    """
    :ivar sents_nb: maximum number of sents to retrieve
    """

    def __init__(self, sents_nb: Union[int, List[int]], **kwargs) -> None:
        self.sents_nb = sents_nb

    def retrieve(
        self, sent_idx: int, entity: NEREntity, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        """Select context for an entity in a sentence in a document

        :param sent_idx: the index of the sentence in the document
        :param document: document in where to find the context
        """
        raise NotImplemented


class RandomContextRetriever(ContextRetriever):
    """A context selector choosing context at random in a document."""

    def retrieve(
        self, sent_idx: int, entity: NEREntity, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        selected_sents_idx = random.sample(
            [i for i in range(len(document)) if not i == sent_idx],
            k=min(len(document) - 1, sents_nb),
        )
        selected_sents_idx = sorted(selected_sents_idx)

        return [
            ContextRetrievalMatch(
                document[i], i, "left" if i < sent_idx else "right", None
            )
            for i in selected_sents_idx
        ]


class SameNounRetriever(ContextRetriever):
    """A context selector that randomly choose a sentence having a
    common name with the current sentence.

    """

    def __init__(self, sents_nb: Union[int, List[int]]):
        """
        :param sents_nb: number of context sentences to select.  If a
            list, the number of context sentences to select will be
            picked randomly among this list at call time.
        """
        # nltk pos tagging dependency
        nltk.download("averaged_perceptron_tagger")
        super().__init__(sents_nb)

    def retrieve(
        self, sent_idx: int, entity: NEREntity, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        sent = document[sent_idx]
        tagged = nltk.pos_tag(sent.tokens)
        name_tokens = set([t[0] for t in tagged if t[1].startswith("NN")])

        # other sentences from the document with at least one token
        # from sent
        selected_sents_idx = [
            i
            for i, s in enumerate(document)
            if not i == sent_idx and len(name_tokens.intersection(set(s.tokens))) > 0
        ]

        # keep at most k sentences
        selected_sents_idx = random.sample(
            selected_sents_idx, k=min(sents_nb, len(selected_sents_idx))
        )
        selected_sents_idx = sorted(selected_sents_idx)

        return [
            ContextRetrievalMatch(
                document[i], i, "left" if i < sent_idx else "right", None
            )
            for i in selected_sents_idx
        ]


class NeighborsContextRetriever(ContextRetriever):
    """A context selector that chooses nearby sentences."""

    def __init__(self, sents_nb: Union[int, List[int]]):
        if isinstance(sents_nb, int):
            assert sents_nb % 2 == 0
        elif isinstance(sents_nb, list):
            assert all([nb % 2 == 0 for nb in sents_nb])

        super().__init__(sents_nb)

    def retrieve(
        self, sent_idx: int, entity: NEREntity, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        left_sents_nb = sents_nb // 2
        right_sents_nb = sents_nb // 2

        left_ctx = [
            ContextRetrievalMatch(document[i], i, "left", None)
            for i in range(max(0, sent_idx - left_sents_nb), sent_idx)
        ]

        right_ctx = [
            ContextRetrievalMatch(document[i], i, "right", None)
            for i in range(
                min(len(document) - 1, sent_idx + 1),
                min(len(document), sent_idx + 1 + right_sents_nb),
            )
        ]

        return left_ctx + right_ctx


class LeftContextRetriever(ContextRetriever):
    """"""

    def __init__(self, sents_nb: Union[int, List[int]]):
        super().__init__(sents_nb)

    def retrieve(
        self, sent_idx: int, entity: NEREntity, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        return [
            ContextRetrievalMatch(document[i], i, "left", None)
            for i in range(max(0, sent_idx - sents_nb), sent_idx)
        ]


class RightContextRetriever(ContextRetriever):
    """"""

    def __init__(self, sents_nb: Union[int, List[int]]):
        super().__init__(sents_nb)

    def retrieve(
        self, sent_idx: int, entity: NEREntity, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        return [
            ContextRetrievalMatch(document[i], i, "right", None)
            for i in range(
                min(len(document) - 1, sent_idx + 1),
                min(len(document), sent_idx + 1 + sents_nb),
            )
        ]


class BM25ContextRetriever(ContextRetriever):
    """A context selector that selects sentences according to BM25 ranking formula."""

    def __init__(self, sents_nb: Union[int, List[int]]) -> None:
        """
        :param sents_nb: number of context sentences to select.  If a
            list, the number of context sentences to select will be
            picked randomly among this list at call time.
        """
        super().__init__(sents_nb)

    @staticmethod
    def _get_bm25_model(document: List[NERSentence]) -> BM25Okapi:
        return BM25Okapi([sent.tokens for sent in document])

    def retrieve(
        self, sent_idx: int, entity: NEREntity, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        bm25_model = BM25ContextRetriever._get_bm25_model(document)
        query = document[sent_idx].tokens
        sent_scores = bm25_model.get_scores(query)
        sent_scores[sent_idx] = -1  # don't retrieve self
        topk_values, topk_indexs = torch.topk(
            torch.tensor(sent_scores), k=min(sents_nb, len(sent_scores)), dim=0
        )
        return [
            ContextRetrievalMatch(
                document[index], index, "left" if index < sent_idx else "right", value
            )
            for value, index in zip(topk_values.tolist(), topk_indexs.tolist())
        ]


def other_sents_with_entity(
    sent_i: int, entity: NEREntity, document: List[NERSentence]
) -> List[Tuple[int, NERSentence]]:
    """
    For a ``document``, returns sentences other than sent at index
    ``sent_i`` that contains an ``entity``.

    :return: a list of tuple of form ``(sent_i, sent)``
    """
    entity_str = " ".join(entity.tokens)
    sents = []
    for i, sent in enumerate(document):
        if i == sent_i:
            continue
        sent_str = " ".join(sent.tokens)
        if entity_str in sent_str:
            sents.append((i, sent))
    return sents


class SameEntityContextRetriever(ContextRetriever):
    def __init__(self, sents_nb: Union[int, List[int]], **kwargs) -> None:
        super().__init__(sents_nb)

    def retrieve(
        self, sent_idx: int, entity: NEREntity, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        other_sents = other_sents_with_entity(sent_idx, entity, document)
        return [
            ContextRetrievalMatch(
                o_sent, o_sent_i, "left" if o_sent_i < sent_idx else "right", None
            )
            for o_sent_i, o_sent in other_sents
        ][:sents_nb]


@dataclass(frozen=True)
class ContextRetrievalExample:
    """A context selection example, to be used for training a context selector."""

    #: sentence on which NER is performed
    sent: List[str]
    #: target entity
    entity: NEREntity
    #: NER tags
    sent_tags: List[str]
    #: context to assist during prediction
    context: List[str]
    #: context NER tags
    context_tags: List[str]
    #: context side (does the context comes from the left or the right of ``sent`` ?)
    context_side: Literal["left", "right"]
    #: usefulness of the exemple, either -1, 0 or 1. between -1 and 1. Can be ``None``
    # when the usefulness is not known.
    usefulness: Optional[Literal[-1, 0, 1]] = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.sent),
                tuple(self.sent_tags),
                tuple(self.context),
                tuple(self.context_tags),
                self.context_side,
                self.usefulness,
            )
        )

    @staticmethod
    def gen_examples(
        sent_i: int,
        entity: NEREntity,
        entity_pred: List[str],
        document: List[NERSentence],
        ner_model,
        batch_size: int,
        tags: Set[str],
    ) -> List[ContextRetrievalExample]:
        # TODO: dev
        # retrieve sents with target entity
        # other_sents = other_sents_with_entity(sent_i, entity, document)
        other_sents = [(i, sent) for i, sent in enumerate(document) if not i == sent_i]

        # prepare NERDataset where retrieved sents are used as context
        sent = document[sent_i]
        sent_with_ctx = [
            NERSentence(
                sent.tokens,
                sent.tags,
                left_context=[other_sent] if other_sent_i < sent_i else [],
                right_context=[other_sent] if other_sent_i > sent_i else [],
            )
            for other_sent_i, other_sent in other_sents
        ]
        dataset = NERDataset([sent_with_ctx], tags)

        # make predictions for sent + other sents as context
        preds_with_other_sents = predict(
            ner_model, dataset, batch_size=batch_size, quiet=True
        )

        # categorize predictions into problematic, useless and useful
        cr_examples = []
        for (other_sent_i, other_sent), pred_with_other_sent in zip(
            other_sents, preds_with_other_sents.tags
        ):
            entity_tags = sent.tags[entity.start_idx : entity.end_idx + 1]
            entity_pred_with_other_sent = pred_with_other_sent[
                entity.start_idx : entity.end_idx + 1
            ]

            entity_pred_was_ok = entity_pred == sent.tags
            entity_pred_is_ok = entity_tags == entity_pred_with_other_sent
            if entity_pred_was_ok:
                usefulness = 0 if entity_pred_is_ok else -1
            else:
                usefulness = 1 if entity_pred_is_ok else 0

            ctx_side = "left" if other_sent_i < sent_i else "right"

            cr_examples.append(
                ContextRetrievalExample(
                    sent.tokens,
                    entity,
                    sent.tags,
                    other_sent.tokens,
                    other_sent.tags,
                    ctx_side,
                    usefulness,
                )
            )

        return cr_examples


class ContextRetrievalDataset(Dataset):
    """"""

    def __init__(
        self,
        examples: List[ContextRetrievalExample],
        tokenizer: Optional[BertTokenizerFast] = None,
    ) -> None:
        self.examples = examples
        if tokenizer is None:
            tokenizer = get_tokenizer()
        self.tokenizer: BertTokenizerFast = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> BatchEncoding:
        """Get a BatchEncoding representing example at index.

        :param index: index of the example to retrieve

        :return: a ``BatchEncoding``, with key ``'label'`` set.
        """
        example = self.examples[index]

        # target entity markers
        sent = copy.copy(example.sent)
        sent.insert(example.entity.end_idx + 1, "[/ENTITY]")
        sent.insert(example.entity.start_idx, "[ENTITY]")

        if example.context_side == "left":
            tokens = example.context + ["<", "[SEP]"] + sent
        elif example.context_side == "right":
            tokens = sent + ["[SEP]", ">"] + example.context
        else:
            raise ValueError

        batch: BatchEncoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
        )

        if not example.usefulness is None:
            # NOTE: example.usefulness is either -1, 0 or 1.
            #       we shift that to class indices {0, 1, 2}
            batch["label"] = example.usefulness + 1

        return batch

    def to_jsonifiable(self) -> List[dict]:
        return [vars(example) for example in self.examples]

    def labels(self) -> Optional[List[float]]:
        if any([ex.usefulness is None for ex in self.examples]):
            return None
        return [ex.usefulness for ex in self.examples]  # type: ignore

    def downsampled(self, downsample_ratio: float = 0.05) -> ContextRetrievalDataset:
        """Downsample class '0' for the dataset

        :param downsample_ratio:
        """
        neg_examples = [e for e in self.examples if e.usefulness == -1]
        pos_examples = [e for e in self.examples if e.usefulness == 1]
        null_examples = [e for e in self.examples if e.usefulness == 0]
        null_examples = null_examples[: int(downsample_ratio * len(null_examples))]
        examples = neg_examples + pos_examples + null_examples
        random.shuffle(examples)
        return ContextRetrievalDataset(examples)

    @staticmethod
    def gen_dataset(
        ner_dataset: NERDataset,
        ner_model: BertForTokenClassification,
        batch_size: int,
    ) -> ContextRetrievalDataset:
        examples = []
        for doc in tqdm(ner_dataset.documents):
            preds = predict(
                ner_model,
                NERDataset([doc], ner_dataset.tags, ner_dataset.tokenizer),
                quiet=True,
            )
            for sent_i, (sent, sent_preds) in enumerate(zip(doc, preds.tags)):
                for entity in entities_from_bio_tags(sent.tokens, sent.tags):
                    entity_pred = sent_preds[entity.start_idx : entity.end_idx + 1]
                    examples += ContextRetrievalExample.gen_examples(
                        sent_i,
                        entity,
                        entity_pred,
                        doc,
                        ner_model,
                        batch_size,
                        ner_dataset.tags,
                    )
        return ContextRetrievalDataset(examples)


class NeuralContextRetriever(ContextRetriever):
    """A context selector powered by BERT"""

    def __init__(
        self,
        pretrained_model: Union[str, BertForSequenceClassification],
        heuristic_context_selector: str,
        heuristic_context_selector_kwargs: Dict[str, Any],
        batch_size: int,
        sents_nb: int,
        use_neg_class: bool = False,
    ) -> None:
        """
        :param pretrained_model_name: pretrained model name, used to
            load a :class:`transformers.BertForSequenceClassification`

        :param heuristic_context_selector: name of the context
            selector to use as retrieval heuristic, from
            ``context_selector_name_to_class``

        :param heuristic_context_selector_kwargs: kwargs to pass the
            heuristic context retriever at instantiation time

        :param batch_size: batch size used at inference

        :param sents_nb: max number of sents to retrieve

        :param use_neg_class:
        """
        if isinstance(pretrained_model, str):
            self.ctx_classifier = BertForSequenceClassification.from_pretrained(
                pretrained_model
            )  # type: ignore
        else:
            self.ctx_classifier = pretrained_model
        self.ctx_classifier = cast(BertForSequenceClassification, self.ctx_classifier)

        self.tokenizer = get_tokenizer()

        selector_class = context_retriever_name_to_class[heuristic_context_selector]
        self.heuristic_context_selector = selector_class(
            **heuristic_context_selector_kwargs
        )

        self.batch_size = batch_size

        self.use_neg_class = use_neg_class

        super().__init__(sents_nb)

    def set_heuristic_sents_nb_(self, sents_nb: int):
        self.heuristic_context_selector.sents_nb = sents_nb

    def predict(
        self,
        dataset: Union[ContextRetrievalDataset, List[ContextRetrievalExample]],
        device_str: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> torch.Tensor:
        """
        :param dataset: A list of :class:`ContextSelectionExample`
        :param device_str: torch device

        :return: A tensor of shape ``(len(dataset), 3)`` of class
                 scores
        """
        if isinstance(dataset, list):
            dataset = ContextRetrievalDataset(dataset, self.tokenizer)

        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        self.ctx_classifier = self.ctx_classifier.to(device)  # type: ignore

        data_collator = DataCollatorWithPadding(dataset.tokenizer)  # type: ignore
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)  # type: ignore

        # inference using self.ctx_classifier
        self.ctx_classifier = self.ctx_classifier.eval()
        with torch.no_grad():
            scores = torch.zeros((0,)).to(device)
            for X in dataloader:
                X = X.to(device)
                # out.logits is of shape (batch_size, 3)
                out = self.ctx_classifier(
                    X["input_ids"],
                    token_type_ids=X["token_type_ids"],
                    attention_mask=X["attention_mask"],
                )
                # (batch_size, 3)
                pred = torch.softmax(out.logits, dim=1)
                scores = torch.cat([scores, pred], dim=0)

        return scores

    def retrieve(
        self, sent_idx: int, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ctx_classifier = self.ctx_classifier.to(device)  # type: ignore

        sent = document[sent_idx]

        # get self.heuristic_retrieval_sents_nb potentially important
        # context sentences
        ctx_matchs = self.heuristic_retrieve_ctx(sent_idx, document)

        # no context retrieved by heuristic : nothing to do
        if len(ctx_matchs) == 0:
            return []

        # prepare datas for inference
        ctx_dataset = [
            ContextRetrievalExample(
                sent.tokens,
                sent.tags,
                ctx_match.sentence.tokens,
                ctx_match.sentence.tags,
                ctx_match.side,
            )
            for ctx_match in ctx_matchs
        ]
        # (len(dataset), 3)
        scores = self.predict(ctx_dataset)

        for i, ctx_match in enumerate(ctx_matchs):
            pos_score = float(scores[i, 2].item())
            if self.use_neg_class:
                neg_score = float(scores[i, 0].item())
                ctx_match.score = pos_score - neg_score
            else:
                ctx_match.score = pos_score

        return sorted(ctx_matchs, key=lambda m: -m.score)[: self.sents_nb]  # type: ignore

    def heuristic_retrieve_ctx(
        self, sent_idx: int, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        """Retrieve potentially useful context sentences to help
        predict sent at index ``sent_idx``.

        :param sent_idx: index of the sent for which NER predictions
            are made.
        :param document: the predicted sentence's document.
        """
        return self.heuristic_context_selector.retrieve(sent_idx, document)

    @staticmethod
    def train_context_selector(
        ctx_dataset: ContextRetrievalDataset,
        epochs_nb: int,
        batch_size: int,
        learning_rate: float,
        _run: Optional[Run] = None,
        log_full_loss: bool = False,
        weights: Optional[torch.Tensor] = None,
        quiet: bool = False,
        valid_dataset: Optional[ContextRetrievalDataset] = None,
        dropout: float = 0.1,
    ) -> BertForSequenceClassification:
        """Instantiate and train a context classifier.

        :param ner_model: an already trained NER model used to
            generate the context selection dataset.
        :param train_dataset: NER dataset used to generate the context
            selection dataset.
        :param epochs_nb: number of training epochs.
        :param batch_size:
        :param weights_bins_nb: number of loss weight bins.  If
            ``None``, the MSELoss will not be weighted.
        :param _run: current sacred run.  If not ``None``, will be
            used to record training metrics.
        :param log_full_loss: if ``True``, log the loss at each batch
            (otherwise, only log mean epochs loss)
        :param weights: :class:`torch.nn.CrossEntropyLoss` weights, oh
            shape ``(3)``.
        :param quiet: if ``True``, no loading bar will be displayed
        :param valid_dataset: a validation dataset, used when logging
            validation metrics using ``_run``

        :return: a trained ``BertForSequenceClassification``
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ctx_classifier = BertForSequenceClassification.from_pretrained(
            "bert-base-cased",
            num_labels=3,
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout,
        )
        ctx_classifier = cast(BertForSequenceClassification, ctx_classifier)
        ctx_classifier = ctx_classifier.to(device)

        if not weights is None:
            weights = weights.to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.AdamW(ctx_classifier.parameters(), lr=learning_rate)

        data_collator = DataCollatorWithPadding(ctx_dataset.tokenizer)  # type: ignore
        dataloader = DataLoader(
            ctx_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
        )

        for _ in range(epochs_nb):

            # * validation metrics
            if not _run is None and not valid_dataset is None:
                valid_dataloader = DataLoader(
                    valid_dataset, batch_size=batch_size, collate_fn=data_collator
                )
                ctx_classifier = ctx_classifier.eval()
                epoch_losses = []
                with torch.no_grad():
                    for X in tqdm(valid_dataloader, disable=quiet):
                        X = X.to(device)
                        out = ctx_classifier(
                            X["input_ids"],
                            token_type_ids=X["token_type_ids"],
                            attention_mask=X["attention_mask"],
                        )
                        loss = loss_fn(out.logits, X["labels"])
                        _run.log_scalar(
                            "neural_selector_training.valid_loss", loss.item()
                        )
                        epoch_losses.append(loss.item())
                _run.log_scalar(
                    "neural_selector_training.mean_epoch_valid_loss",
                    sum(epoch_losses) / len(epoch_losses),
                )

            # * train
            epoch_losses = []
            epoch_preds = []
            epoch_labels = []

            ctx_classifier = ctx_classifier.train()

            data_tqdm = tqdm(dataloader, disable=quiet)
            for X in data_tqdm:

                optimizer.zero_grad()

                X = X.to(device)

                out = ctx_classifier(
                    X["input_ids"],
                    token_type_ids=X["token_type_ids"],
                    attention_mask=X["attention_mask"],
                )
                loss = loss_fn(out.logits, X["labels"])
                loss.backward()

                optimizer.step()

                if not _run is None and log_full_loss:
                    _run.log_scalar("neural_selector_training.loss", loss.item())

                data_tqdm.set_description(f"loss : {loss.item():.3f}")
                epoch_losses.append(loss.item())

                epoch_preds += torch.argmax(out.logits, dim=1).tolist()
                epoch_labels += X["labels"].tolist()

            # ** train metrics logging
            mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            tqdm.write(f"epoch mean loss : {mean_epoch_loss:.3f}")
            if not _run is None:
                # mean loss
                _run.log_scalar(
                    "neural_selector_training.mean_epoch_loss", mean_epoch_loss
                )
                # metrics
                # TODO:
                p, r, f, _ = precision_recall_fscore_support(
                    epoch_labels, epoch_preds, average="micro"
                )
                _run.log_scalar("neural_selector_training.epoch_precision", p)
                _run.log_scalar("neural_selector_training.epoch_recall", r)
                _run.log_scalar("neural_selector_training.epoch_f1", f)

        return ctx_classifier


class OracleContextRetriever(ContextRetriever):
    """
    A context retriever that always return the ``sents_nb`` most
    helpful contexts retrieved by its ``preliminary_ctx_selector``
    according to its given ``ner_model``
    """

    def __init__(
        self,
        sents_nb: Union[int, List[int]],
        preliminary_ctx_selector: ContextRetriever,
        ner_model: BertForTokenClassification,
        batch_size: int,
        tags: Set[str],
    ) -> None:
        self.preliminary_ctx_selector = preliminary_ctx_selector
        self.ner_model = ner_model
        self.batch_size = batch_size
        self.tags = tags
        super().__init__(sents_nb)

    def set_heuristic_sents_nb_(self, sents_nb: int):
        self.preliminary_ctx_selector.sents_nb = sents_nb

    @staticmethod
    def _pred_error(sent: NERSentence, pred: List[str]) -> int:
        """Compute error between a reference sentence and a prediction
        :param sent: reference sentence
        :param preds: list of tags of shape ``(sentence_size)``
        :param tag_to_id: a mapping from a tag to its id in the
            vocabulary.
        :return: the number of incorrect tags in ``preds``
        """
        assert len(pred) == len(sent.tags)
        return sum([1 if t != p else 0 for p, t in zip(pred, sent.tags)])

    def retrieve(
        self, sent_idx: int, entity: NEREntity, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        sent = document[sent_idx]
        preds = predict(
            self.ner_model, NERDataset([[sent]], self.tags), quiet=True, batch_size=1
        )

        contexts = self.preliminary_ctx_selector.retrieve(sent_idx, entity, document)

        sent_with_ctx = sent_with_ctx_from_matchs(sent, contexts)

        ctx_preds = predict(
            self.ner_model,
            NERDataset([sent_with_ctx], self.tags),
            quiet=True,
            batch_size=self.batch_size,
        )

        for context, ctx_pred in zip(contexts, ctx_preds.tags):
            ctx_err = OracleContextRetriever._pred_error(sent, ctx_pred)
            err = OracleContextRetriever._pred_error(sent, preds.tags[0])
            if ctx_err > err:
                context.score = -1
            elif ctx_err < err:
                context.score = 1
            else:
                context.score = 0

        return sorted(contexts, key=lambda c: -c.score)[:sents_nb]  # type: ignore


class AllContextRetriever(ContextRetriever):
    """A stub context retriever that retrieves _every_ sentence"""

    def __init__(self, sents_nb: Union[int, List[int]]) -> None:
        """'
        .. warning::

            ``sents_nb`` is *ignored*
        """
        super().__init__(sents_nb)

    def retrieve(
        self, sent_idx: int, entity: NEREntity, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        matchs = []
        for sent_i, sent in enumerate(document):
            if sent_i == sent_idx:
                continue
            matchs.append(
                ContextRetrievalMatch(
                    sent, sent_i, "left" if sent_i < sent_idx else "right", None
                )
            )
        return matchs


context_retriever_name_to_class: Dict[str, Type[ContextRetriever]] = {
    "neural": NeuralContextRetriever,
    "neighbors": NeighborsContextRetriever,
    "left": LeftContextRetriever,
    "right": RightContextRetriever,
    "bm25": BM25ContextRetriever,
    "samenoun": SameNounRetriever,
    "sameentity": SameEntityContextRetriever,
    "random": RandomContextRetriever,
    "all": AllContextRetriever,
}
