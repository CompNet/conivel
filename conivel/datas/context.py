from __future__ import annotations
from typing import Any, Dict, Generator, List, Literal, Optional, Set, Type, Union, cast
import random, json, os
from dataclasses import dataclass, field
import nltk
from sacred.run import Run
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForTokenClassification, BertForSequenceClassification, BertTokenizerFast, DataCollatorWithPadding  # type: ignore
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from rank_bm25 import BM25Okapi
from more_itertools import flatten
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset
from conivel.utils import (
    flattened,
    get_tokenizer,
)
from conivel.predict import predict


@dataclass
class ContextRetrievalMatch:
    sentence: NERSentence
    sentence_idx: int
    side: Literal["left", "right"]
    score: Optional[float]
    #: additional annotations for debug purposes
    _custom_annotations: Dict[str, Any] = field(default_factory=lambda: dict())

    def to_jsonifiable(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence.to_jsonifiable(),
            "sentence_idx": self.sentence_idx,
            "side": self.side,
            "score": self.score,
            "_custom_annotations": self._custom_annotations,
        }

    @staticmethod
    def from_jsonifiable(j: Dict[str, Any]) -> ContextRetrievalMatch:
        return ContextRetrievalMatch(
            NERSentence.from_jsonifiable(j["sentence"]),
            j["sentence_idx"],
            j["side"],
            j["score"],
            j["_custom_annotations"],
        )


class ContextRetriever:
    """
    :ivar sents_nb: maximum number of sents to retrieve
    """

    def __init__(self, sents_nb: Union[int, List[int]], **kwargs) -> None:
        self.sents_nb = sents_nb

    def __call__(
        self,
        dataset: NERDataset,
        quiet: bool = True,
        extended_documents: Optional[List[List[NERSentence]]] = None,
    ) -> NERDataset:
        """Retrieve context for each sentence of a :class:`NERDataset`

        :param dataset:
        :param quiet:
        :param extended_documents: Extended version of
            ``dataset.documents`` for retrieval purposes.
        """
        if extended_documents:
            assert len(extended_documents) == len(dataset.documents)

        new_docs = []

        for doc_i, document in enumerate(tqdm(dataset.documents, disable=quiet)):
            new_doc = []

            retrieval_doc = (
                extended_documents[doc_i] if extended_documents else document
            )

            for sent_i, sent in enumerate(document):
                retrieval_matchs = sorted(
                    self.retrieve(sent_i, retrieval_doc), key=lambda m: m.sentence_idx
                )

                new_doc.append(
                    NERSentence(
                        sent.tokens,
                        sent.tags,
                        [m.sentence for m in retrieval_matchs if m.side == "left"],
                        [m.sentence for m in retrieval_matchs if m.side == "right"],
                        # put retrieved matchs for debug purposes
                        _custom_annotations={
                            "matchs": [m.to_jsonifiable() for m in retrieval_matchs]
                        },
                    )
                )

            new_docs.append(new_doc)

        return NERDataset(new_docs, tags=dataset.tags, tokenizer=dataset.tokenizer)

    def dataset_with_contexts(
        self,
        dataset: NERDataset,
        sents_nb_list: List[int],
        quiet: bool = True,
        extended_documents: Optional[List[List[NERSentence]]] = None,
    ) -> Generator[NERDataset, None, None]:
        """
        retrieve context for each sentence of the given
        :class:`NERDataset`.  Yield a dataset with ``k`` context
        sentences for each integer of ``sents_nb_list``.  Notably,
        this function only retrieve context once (for
        ``max(sents_nb_list)``) for optimisation purposes.

        .. note::

            ``self.sents_nb`` must be equal to ``sents_nb_list``

        :param dataset:
        :param sents_nb_list:
        :param quiet:
        :param extended_documents:
        """
        assert self.sents_nb == max(sents_nb_list)
        if extended_documents:
            assert len(extended_documents) == len(dataset.documents)

        matchs: List[List[List[ContextRetrievalMatch]]] = []
        for doc_i, doc in enumerate(tqdm(dataset.documents, disable=quiet)):
            doc_matchs = []
            retrieval_doc = extended_documents[doc_i] if extended_documents else doc
            for sent_i, sent in enumerate(doc):
                doc_matchs.append(self.retrieve(sent_i, retrieval_doc))
            matchs.append(doc_matchs)

        for sents_nb in sorted(sents_nb_list):
            docs = []
            for doc, doc_matchs in zip(dataset.documents, matchs):
                sents = []
                for sent, sent_matchs in zip(doc, doc_matchs):
                    l_ctx = [
                        m.sentence for m in sent_matchs[:sents_nb] if m.side == "left"
                    ]
                    r_ctx = [
                        m.sentence for m in sent_matchs[:sents_nb] if m.side == "right"
                    ]
                    sent_with_ctx = NERSentence(
                        sent.tokens,
                        sent.tags,
                        left_context=l_ctx,
                        right_context=r_ctx,
                        _custom_annotations={
                            "matchs": [
                                m.to_jsonifiable() for m in sent_matchs[:sents_nb]
                            ]
                        },
                    )
                    sents.append(sent_with_ctx)
                docs.append(sents)
            yield NERDataset(docs, dataset.tags, dataset.tokenizer)

    def retrieve(
        self, sent_idx: int, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        """Select context for a sentence in a document

        :param sent_idx: the index of the sentence in the document
        :param document: document in where to find the context
        """
        raise NotImplemented


def sent_with_ctx_from_matchs(
    sent: NERSentence, ctx_matchs: List[ContextRetrievalMatch]
) -> List[NERSentence]:
    return [
        NERSentence(
            sent.tokens,
            sent.tags,
            left_context=[ctx_match.sentence] if ctx_match.side == "left" else [],
            right_context=[ctx_match.sentence] if ctx_match.side == "right" else [],
        )
        for ctx_match in ctx_matchs
    ]


class RandomContextRetriever(ContextRetriever):
    """A context selector choosing context at random in a document."""

    def retrieve(
        self, sent_idx: int, document: List[NERSentence]
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
        self, sent_idx: int, document: List[NERSentence]
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
        self, sent_idx: int, document: List[NERSentence]
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
        self, sent_idx: int, document: List[NERSentence]
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
        self, sent_idx: int, document: List[NERSentence]
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
        self, sent_idx: int, document: List[NERSentence]
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


class BM25RestrictedContextRetriever(ContextRetriever):
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
        self, sent_idx: int, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        bm25_model = BM25ContextRetriever._get_bm25_model(document)
        query = document[sent_idx].tokens
        sent_scores = bm25_model.get_scores(query)
        sent_scores[sent_idx] = -1  # don't retrieve self
        # HACK: exclude close sentences
        for i in range(1, 7):
            try:
                sent_scores[sent_idx + i] = -1
            except IndexError:
                pass
            if sent_idx - i > 0:
                sent_scores[sent_idx - i] = -1
        topk_values, topk_indexs = torch.topk(
            torch.tensor(sent_scores), k=min(sents_nb, len(sent_scores)), dim=0
        )
        return [
            ContextRetrievalMatch(
                document[index], index, "left" if index < sent_idx else "right", value
            )
            for value, index in zip(topk_values.tolist(), topk_indexs.tolist())
        ]


@dataclass(frozen=True)
class ContextRetrievalExample:
    """A context selection example, to be used for training a context selector."""

    #: sentence on which NER is performed
    sent: List[str]
    #: NER tags
    sent_tags: List[str]
    #: context to assist during prediction
    context: List[str]
    #: context NER tags
    context_tags: List[str]
    #: context side (doest the context comes from the left or the right of ``sent`` ?)
    context_side: Literal["left", "right"]
    #: usefulness of the exemple, either 0 or 1. Can be ``None`` for an unlabeled example
    usefulness: Optional[Literal[0, 1]] = None

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

        tokens = example.context + ["[SEP]"] + example.sent

        batch: BatchEncoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
        )

        if not example.usefulness is None:
            batch["label"] = example.usefulness

        return batch

    @staticmethod
    def concatenated(
        datasets: List[ContextRetrievalDataset],
    ) -> ContextRetrievalDataset:
        assert len(datasets) > 0
        return ContextRetrievalDataset(
            list(flatten([d.examples for d in datasets])), datasets[0].tokenizer
        )

    def to_jsonifiable(self) -> List[dict]:
        return [vars(example) for example in self.examples]

    def labels(self) -> Optional[List[float]]:
        if any([ex.usefulness is None for ex in self.examples]):
            return None
        return [ex.usefulness for ex in self.examples]  # type: ignore


class NeuralContextRetriever(ContextRetriever):
    """A context selector powered by BERT"""

    def __init__(
        self,
        pretrained_model: Union[str, BertForSequenceClassification],
        heuristic_context_selector: ContextRetriever,
        batch_size: int,
        sents_nb: int,
        threshold: float = 0.0,
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

        self.heuristic_context_selector = heuristic_context_selector

        self.batch_size = batch_size

        self.threshold = threshold

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

        :return: A tensor of shape ``(len(dataset), 2)`` of class
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
                # out.logits is of shape (batch_size, 2)
                out = self.ctx_classifier(
                    X["input_ids"],
                    token_type_ids=X["token_type_ids"],
                    attention_mask=X["attention_mask"],
                )
                # (batch_size, 2)
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
        # (len(dataset), 2)
        scores = self.predict(ctx_dataset)

        for i, ctx_match in enumerate(ctx_matchs):
            ctx_match.score = float(scores[i, 1].item())

        assert all([not m.score is None for m in ctx_matchs])
        return [
            m
            for m in sorted(ctx_matchs, key=lambda m: -m.score)[: self.sents_nb]  # type: ignore
            if m.score > self.threshold  # type: ignore
        ]

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
    def _pred_error(
        sent_tags: List[str], pred_scores: torch.Tensor, tag2id: Dict[str, int]
    ) -> float:
        """Compute error between a reference sentence and a prediction

        :param sent: reference sentence
        :param pred_scores: score for each tag class, of shape
            ``(sentence_size, vocab_size)``
        :param tag2id: a mapping from a tag to its id in the
            vocabulary.

        :return: the number of incorrect tags in ``preds``
        """
        assert len(sent_tags) == pred_scores.shape[0]
        return sum(
            [
                1.0 - float(pred_scores[i][tag2id[tag]].item())
                for i, tag in enumerate(sent_tags)
            ]
        )

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
        huggingface_id: str = "bert-base-cased",
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
            shape ``(2)``.
        :param quiet: if ``True``, no loading bar will be displayed
        :param valid_dataset: a validation dataset, used when logging
            validation metrics using ``_run``

        :return: a trained ``BertForSequenceClassification``
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ctx_classifier = BertForSequenceClassification.from_pretrained(
            huggingface_id,
            num_labels=2,
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


class CombinedContextRetriever(ContextRetriever):
    def __init__(
        self, sents_nb: Union[int, List[int]], retrievers: List[ContextRetriever]
    ) -> None:
        self.retrievers = retrievers
        super().__init__(sents_nb)

    def retrieve(
        self, sent_idx: int, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        # retrieve match for all retrievers
        matchs = flattened([r.retrieve(sent_idx, document) for r in self.retrievers])

        uniq_matchs_sents = set()
        uniq_matchs = []
        for m in matchs:
            if not m.sentence in uniq_matchs_sents:
                uniq_matchs_sents.add(m.sentence)
                uniq_matchs.append(m)

        # it's not possible to compare scores from different
        # retrievers. Therefore, we do not make any assumption on the
        # order of the returned matchs.
        random.shuffle(uniq_matchs)

        return uniq_matchs[:sents_nb]


class OracleNeuralContextRetriever(ContextRetriever):
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
        inverted: bool = False,
    ) -> None:
        self.preliminary_ctx_selector = preliminary_ctx_selector
        self.ner_model = ner_model
        self.batch_size = batch_size
        self.tags = tags
        self.tag_to_id: Dict[str, int] = {
            tag: i for i, tag in enumerate(sorted(list(self.tags)))
        }
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.inverted = inverted
        super().__init__(sents_nb)

    def set_heuristic_sents_nb_(self, sents_nb: int):
        self.preliminary_ctx_selector.sents_nb = sents_nb

    def retrieve(
        self, sent_idx: int, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        sent = document[sent_idx]
        p = predict(
            self.ner_model,
            NERDataset([[sent]], self.tags),
            quiet=True,
            batch_size=1,
            additional_outputs={"scores"},
        )
        assert not p.scores is None
        pred_scores = p.scores[0]
        pred_tags = p.tags[0]

        contexts = self.preliminary_ctx_selector.retrieve(sent_idx, document)

        sent_with_ctx = sent_with_ctx_from_matchs(sent, contexts)

        ctx_preds = predict(
            self.ner_model,
            NERDataset([sent_with_ctx], self.tags),
            quiet=True,
            batch_size=self.batch_size,
            additional_outputs={"scores"},
        )
        assert not ctx_preds.scores is None

        err = NeuralContextRetriever._pred_error(sent.tags, pred_scores, self.tag_to_id)
        for context, ctx_pred, ctx_scores in zip(
            contexts, ctx_preds.tags, ctx_preds.scores
        ):
            # annotate context score
            ctx_err = NeuralContextRetriever._pred_error(
                sent.tags, ctx_scores, self.tag_to_id
            )
            context.score = err - ctx_err
            # annotate custom debug attributes
            context._custom_annotations["pred_tags_no_ctx"] = pred_tags
            context._custom_annotations["pred_tags_with_ctx"] = ctx_pred
            context._custom_annotations["err"] = err
            context._custom_annotations["ctx_err"] = ctx_err

        return sorted(contexts, key=lambda c: c.score if self.inverted else -c.score)[:sents_nb]  # type: ignore


class AllContextRetriever(ContextRetriever):
    """A stub context retriever that retrieves _every_ sentence"""

    def __init__(self, sents_nb: Union[int, List[int]]) -> None:
        """'
        .. warning::
            ``sents_nb`` is *ignored*
        """
        super().__init__(sents_nb)

    def retrieve(
        self, sent_idx: int, document: List[NERSentence]
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


class MonoContextRetriever(ContextRetriever):
    """Class for MonoBERT and MonoT5 retriever."""

    def __init__(
        self,
        sents_nb: Union[int, List[int]],
        heuristic_context_selector: ContextRetriever,
        ranker_class: Union[
            Type["pygaggle.rerank.transformer.MonoBERT"],
            Type["pygaggle.rerank.transformer.MonoT5"],
        ],
    ) -> None:
        from pygaggle.rerank.transformer import MonoBERT, MonoT5
        from transformers import T5ForConditionalGeneration

        if ranker_class == MonoT5:
            model = T5ForConditionalGeneration.from_pretrained(
                "castorini/monot5-base-msmarco-10k"
            )
            self.ranker = MonoT5(model=model)
        else:
            self.ranker = ranker_class()

        self.heuristic_context_selector = heuristic_context_selector

        super().__init__(sents_nb)

    def predict(
        self, sent: NERSentence, matchs: List[ContextRetrievalMatch]
    ) -> List[float]:
        from pygaggle.rerank.base import Query, Text

        matchs_texts = [" ".join(m.sentence.tokens) for m in matchs]
        out = self.ranker.rerank(
            Query(" ".join(sent.tokens)), [Text(m) for m in matchs_texts]
        )

        scores = []
        out_texts = [t.text for t in out]
        for m, m_text in zip(matchs, matchs_texts):
            out_i = out_texts.index(m_text)
            scores.append(out[out_i].score)

        return scores

    def retrieve(
        self, sent_idx: int, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        sent = document[sent_idx]

        matchs = self.heuristic_context_selector.retrieve(sent_idx, document)
        matchs = [m for m in matchs if not m.sentence == sent]
        if len(matchs) == 0:
            return []

        scores = self.predict(sent, matchs)

        for score, m in zip(scores, matchs):
            m.score = score

        return [m for m in sorted(matchs, key=lambda m: -m.score)[:sents_nb]]  # type: ignore


context_retriever_name_to_class: Dict[str, Type[ContextRetriever]] = {
    "neural": NeuralContextRetriever,
    "neighbors": NeighborsContextRetriever,
    "left": LeftContextRetriever,
    "right": RightContextRetriever,
    "bm25": BM25ContextRetriever,
    "bm25_restricted": BM25RestrictedContextRetriever,
    "samenoun": SameNounRetriever,
    "random": RandomContextRetriever,
    "all": AllContextRetriever,
}
