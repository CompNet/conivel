from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Type, Union, cast
import random
from dataclasses import dataclass
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
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset
from conivel.utils import get_tokenizer
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

    def __call__(self, dataset: NERDataset, silent: bool = True) -> NERDataset:
        """retrieve context for each sentence of a :class:`NERDataset`"""
        new_docs = []
        for document in tqdm(dataset.documents, disable=silent):
            new_doc = []
            for sent_i, sent in enumerate(document):
                retrieval_matchs = self.retrieve(sent_i, document)
                new_doc.append(
                    NERSentence(
                        sent.tokens,
                        sent.tags,
                        [m.sentence for m in retrieval_matchs if m.side == "left"],
                        [m.sentence for m in retrieval_matchs if m.side == "right"],
                    )
                )
            new_docs.append(new_doc)
        return NERDataset(new_docs, tags=dataset.tags, tokenizer=dataset.tokenizer)

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
        """
        :param left_sents_nb: number of left context sentences to select
        :param right_sents_nb: number of right context sentences to select
        """
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
            for i in range(sent_idx + 1, sent_idx + 1 + right_sents_nb)
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
            for i in range(sent_idx + 1, sent_idx + 1 + sents_nb)
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
            torch.tensor(sent_scores), k=sents_nb, dim=0
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
    #: context to assist during prediction
    context: List[str]
    #: context side (doest the context comes from the left or the right of ``sent`` ?)
    context_side: Literal["left", "right"]
    #: usefulness of the exemple, either -1, 0 or 1. between -1 and 1. Can be ``None``
    # when the usefulness is not known.
    usefulness: Optional[Literal[-1, 0, 1]] = None
    #: wether the prediction for the ``sent`` of this example was
    # correct or not before applying ``context``. Is ``None`` when not
    # applicable.
    sent_was_correctly_predicted: Optional[bool] = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.sent),
                tuple(self.context),
                self.context_side,
                self.usefulness,
                self.sent_was_correctly_predicted,
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

        if example.context_side == "left":
            tokens = example.context + ["<"] + example.sent
        elif example.context_side == "right":
            tokens = example.sent + [">"] + example.context
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


class NeuralContextRetriever(ContextRetriever):
    """A context selector powered by BERT"""

    def __init__(
        self,
        pretrained_model: Union[str, BertForSequenceClassification],
        heuristic_context_selector: str,
        heuristic_context_selector_kwargs: Dict[str, Any],
        batch_size: int,
        sents_nb: int,
        use_cache: bool = False,
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

        :param use_cache: if ``True``,
            :func:`NeuralContextRetriever.predict` will use an
            internal cache to speed up computations.
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

        self._predict_cache = {}
        self.use_cache = use_cache

        super().__init__(sents_nb)

    def clear_predict_cache_(self):
        """Clear the prediction cache"""
        self._predict_cache = {}

    def _predict_cache_get(self, x: ContextRetrievalExample) -> Optional[float]:
        return self._predict_cache.get(x)

    def _predict_cache_register_(self, x: ContextRetrievalExample, score: float):
        self._predict_cache[x] = score

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
                sent.tokens, ctx_match.sentence.tokens, ctx_match.side, None
            )
            for ctx_match in ctx_matchs
        ]
        # (len(dataset), 3)
        scores = self.predict(ctx_dataset)

        # indices for examples with predicted class 1
        selected_indices = torch.arange(0, scores.shape[0])[
            torch.argmax(scores, dim=1) == 2
        ].tolist()
        # only keep selected matchs
        ctx_matchs = [ctx_matchs[i] for i in selected_indices]
        # assign new score
        for i, match in zip(selected_indices, ctx_matchs):
            match.score = scores[i, 2]

        return sorted(ctx_matchs, key=lambda m: -m.score)[: self.sents_nb]

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

    @staticmethod
    def generate_context_dataset(
        ner_model: BertForTokenClassification,
        train_dataset: NERDataset,
        batch_size: int,
        heuristic_context_selector: str,
        heuristic_context_selector_kwargs: Dict[str, Any],
        _run: Optional[Run] = None,
    ) -> ContextRetrievalDataset:
        """Generate a context selection training dataset.

        The process is as follows :

            1. Make predictions for a NER dataset using an already
               trained NER model.

            2. For each prediction, sample a bunch of possible context
               sentences using some heuristic, and retry predictions
               with those context for the sentence.  Then, the
               difference of errors between the prediction without
               context and the prediction with context is used to
               create a sample of context retrieval.

        .. note::

            For now, uses ``SameWordSelector`` as sampling heuristic.

        :todo: make a choice on heuristic

        :param ner_model: an already trained NER model used to
            generate initial predictions
        :param train_dataset: NER dataset used to extract examples
        :param batch_size: batch size used for NER inference
        :param heuristic_context_selector: name of the context
            selector to use as retrieval heuristic, from
            ``context_selector_name_to_class``
        :param heuristic_context_selector_kwargs: kwargs to pass the
            heuristic context retriever at instantiation time
        :param _run: The current sacred run.  If not ``None``, will be
            used to record generation metrics.

        :return: a ``ContextSelectionDataset`` that can be used to
                 train a context selector.
        """
        preds = predict(ner_model, train_dataset, batch_size=batch_size)

        ctx_selector_class = context_retriever_name_to_class[heuristic_context_selector]
        preliminary_ctx_selector = ctx_selector_class(
            **heuristic_context_selector_kwargs
        )

        ctx_selection_examples = []
        for sent_i, (sent, pred_tags) in tqdm(
            enumerate(zip(train_dataset.sents(), preds.tags)),
            total=len(preds.tags),
        ):
            document = train_dataset.document_for_sent(sent_i)

            pred_error = NeuralContextRetriever._pred_error(sent, pred_tags)

            # retrieve n context sentences
            index_in_doc = train_dataset.sent_document_index(sent_i)
            ctx_matchs = preliminary_ctx_selector.retrieve(index_in_doc, document)
            ctx_sents = sent_with_ctx_from_matchs(sent, ctx_matchs)

            # generate examples by making new predictions with context
            # sentences
            preds_ctx = predict(
                ner_model,
                NERDataset(
                    [ctx_sents],
                    train_dataset.tags,
                    tokenizer=train_dataset.tokenizer,
                ),
                quiet=True,
                batch_size=batch_size,
            )

            for ctx_match, ctx_pred in zip(ctx_matchs, preds_ctx.tags):
                pred_ctx_error = NeuralContextRetriever._pred_error(sent, ctx_pred)
                if pred_ctx_error > pred_error:
                    usefulness = -1
                elif pred_ctx_error < pred_error:
                    usefulness = 1
                else:
                    usefulness = 0
                ctx_selection_examples.append(
                    ContextRetrievalExample(
                        sent.tokens,
                        ctx_match.sentence.tokens,
                        ctx_match.side,
                        usefulness,
                        sent.tags == ctx_pred,
                    )
                )

        # logging
        if not _run is None:
            _run.log_scalar(
                "context_dataset_generation.examples_nb", len(ctx_selection_examples)
            )

            for ex in ctx_selection_examples:
                _run.log_scalar("context_dataset_generation.usefulness", ex.usefulness)

        return ContextRetrievalDataset(ctx_selection_examples)

    @staticmethod
    def train_context_selector(
        ctx_dataset: ContextRetrievalDataset,
        epochs_nb: int,
        batch_size: int,
        learning_rate: float,
        _run: Optional[Run] = None,
        log_full_loss: bool = False,
        weights: Optional[torch.Tensor] = None,
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
        :param weightd: :class:`torch.nn.CrossEntropyLoss` weights, oh
            shape ``(3)``.

        :return: a trained ``BertForSequenceClassification``
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ctx_classifier = BertForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=3
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

            epoch_losses = []
            epoch_preds = []
            epoch_labels = []
            ctx_classifier = ctx_classifier.train()

            data_tqdm = tqdm(dataloader)
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


class IdealNeuralContextRetriever(ContextRetriever):
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
    ) -> None:
        self.preliminary_ctx_selector = preliminary_ctx_selector
        self.ner_model = ner_model
        self.batch_size = batch_size
        super().__init__(sents_nb)

    def set_heuristic_sents_nb_(self, sents_nb: int):
        self.preliminary_ctx_selector.sents_nb = sents_nb

    def retrieve(
        self, sent_idx: int, document: List[NERSentence]
    ) -> List[ContextRetrievalMatch]:
        if isinstance((sents_nb := self.sents_nb), list):
            sents_nb = random.choice(sents_nb)

        contexts = self.preliminary_ctx_selector.retrieve(sent_idx, document)

        sent = document[sent_idx]
        sent_with_ctx = sent_with_ctx_from_matchs(sent, contexts)

        tags = {"O", "B-PER", "I-PER"}
        tag_to_id = {tag: i for i, tag in enumerate(sorted(tags))}

        ctx_preds = predict(
            self.ner_model,
            NERDataset([sent_with_ctx], tags),
            quiet=True,
            batch_size=self.batch_size,
            additional_outputs={"scores"},
        )
        assert not ctx_preds.scores is None

        context_and_err = [
            (
                context,
                NeuralContextRetriever._pred_error(sent, scores, tag_to_id),
            )
            for context, scores in zip(contexts, ctx_preds.scores)
        ]

        ok_contexts_and_err = list(sorted(context_and_err, key=lambda cd: cd[1]))[
            :sents_nb
        ]
        ok_contexts = [context for context, _ in ok_contexts_and_err]

        return ok_contexts


context_retriever_name_to_class: Dict[str, Type[ContextRetriever]] = {
    "neural": NeuralContextRetriever,
    "neighbors": NeighborsContextRetriever,
    "left": LeftContextRetriever,
    "right": RightContextRetriever,
    "bm25": BM25ContextRetriever,
    "samenoun": SameNounRetriever,
    "random": RandomContextRetriever,
}
