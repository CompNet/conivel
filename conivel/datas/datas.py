from __future__ import annotations
from inspect import classify_class_attrs
import random
from typing import List, Dict, Literal, Set, Tuple, Union, Optional, cast
from collections import defaultdict
from dataclasses import dataclass, field

from itertools import chain
from more_itertools import windowed
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
import nltk

from conivel.utils import flattened, get_tokenizer


@dataclass(frozen=True)
class NERSentence:
    tokens: List[str]
    tags: List[str]
    left_context: List[NERSentence] = field(default_factory=lambda: [])
    right_context: List[NERSentence] = field(default_factory=lambda: [])

    def __len__(self) -> int:
        assert len(self.tokens) == len(self.tags)
        return len(self.tokens)

    def __repr__(self) -> str:
        rep = f"(tokens={self.tokens}, tags={self.tags}"
        if len(self.left_context) > 0:
            rep += f", left_context={self.left_context}"
        if len(self.right_context) > 0:
            rep += f", right_context={self.right_context}"
        rep += ")"
        return rep

    def __str__(self) -> str:
        return repr(self)

    @staticmethod
    def sents_with_surrounding_context(
        sents: List[NERSentence], context_size: int = 1
    ) -> List[NERSentence]:
        """Set the context of each sent of the given list, according to surrounding sentence

        :param sents:

        :param context_size: number of surrounding sents to take into account, left and right.
            (a value of 1 means taking one sentence left for ``left_context`` and one sentence
            right for ``right_context``.)

        :return: a list of sentences, with ``left_context`` and ``right_context`` set with
            surrounding sentences.
        """
        new_sents: List[NERSentence] = []

        window_size = 1 + 2 * context_size
        padding = [None] * context_size
        for window_sents in windowed(chain(padding, sents, padding), window_size):
            center_idx = window_size // 2
            center_sent = window_sents[center_idx]
            assert not center_sent is None
            left_ctx = [s for s in window_sents[:center_idx] if not s is None]
            right_ctx = [s for s in window_sents[center_idx + 1 :] if not s is None]
            new_sents.append(
                NERSentence(
                    center_sent.tokens,
                    center_sent.tags,
                    left_context=left_ctx,
                    right_context=right_ctx,
                )
            )

        return new_sents


def batch_to_device(batch: BatchEncoding, device: torch.device) -> BatchEncoding:
    """Send a batch to a torch device, even when containing non-tensor variables"""
    if isinstance(batch, BatchEncoding) and all(
        [isinstance(v, torch.Tensor) for v in batch.values()]
    ):
        return batch.to(device)
    return BatchEncoding(
        {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        },
        encoding=batch.encodings,
    )


def align_tokens_labels_(
    batch_encoding: BatchEncoding, labels: List[str], all_labels: Dict[str, int]
) -> BatchEncoding:
    """Modify a huggingface single batch encoding by adding tokens labels, taking wordpiece into account

    .. note::

        Adapted from https://huggingface.co/docs/transformers/custom_datasets#tok_ner

    :param batch_encoding: a ``'labels'`` key will be added. It must be a single batch.
    :param labels: list of per-token labels. ``None`` labels will be given -100 label,
        in order to be ignored by torch loss functions.
    :param all_labels: mapping of a label to its index
    :return: the modified batch encoding
    """
    labels_ids: List[int] = []
    word_ids = batch_encoding.word_ids(batch_index=0)
    for word_idx in word_ids:
        if word_idx is None:
            labels_ids.append(-100)
            continue
        if labels[word_idx] is None:
            labels_ids.append(-100)
            continue
        labels_ids.append(all_labels[labels[word_idx]])
    batch_encoding["labels"] = labels_ids
    return batch_encoding


def truncate_batch(
    batch: BatchEncoding, direction: Literal["left", "right"], max_length: int = 512
) -> BatchEncoding:
    """Truncate a ``BatchEncoding``. Supports left truncation."""

    if direction == "right":
        batch = BatchEncoding(
            {k: v[:-1][: max_length - 1] + v[-1:] for k, v in batch.items()},
            encoding=batch.encodings,
        )
    else:
        batch = BatchEncoding(
            {k: v[:1] + v[1:][-(max_length - 1) :] for k, v in batch.items()},
            encoding=batch.encodings,
        )

    return batch


class DataCollatorForTokenClassificationWithBatchEncoding:
    """Same as ``transformers.DataCollatorForTokenClassification``, except it :

    - correctly returns a ``BatchEncoding`` object with correct ``encodings``
        attribute.
    - wont try to convert the key ``'tokens_labels_mask'`` that is used to
        determine

    Don't know why this is not the default ?
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = -100

    def __call__(self, features) -> Union[dict, BatchEncoding]:
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        batch = self.tokenizer.pad(
            features,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )
        # keep encodings info dammit
        batch._encodings = [f.encodings[0] for f in features]

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        # ignore "tokens_labels_mask"
        return BatchEncoding(
            {
                k: torch.tensor(v, dtype=torch.int64)
                if not k in {"tokens_labels_mask"}
                else v
                for k, v in batch.items()
            },
            encoding=batch.encodings,
        )


class ContextSelector:
    """"""

    def __init__(self, **kwargs) -> None:
        raise NotImplemented

    def __call__(
        self, sent_idx: int, document: List[NERSentence]
    ) -> Tuple[List[NERSentence], List[NERSentence]]:
        """Select context for a sentence in a document

        :param sent_idx: the index of the sentence in the document
        :param document: document in where to find the context

        :return: a tuple with the left and right context of the input
            sent
        """
        raise NotImplemented


class RandomContextSelector(ContextSelector):
    """A context selector choosing context at random in a document."""

    def __init__(self, sents_nb):
        """
        :param sents_nb: number of context sentences to select
        """
        self.sents_nb = sents_nb

    def __call__(
        self, sent_idx: int, document: List[NERSentence]
    ) -> Tuple[List[NERSentence], List[NERSentence]]:
        """ """
        selected_sents_idx = random.sample(
            [i for i in range(len(document)) if not i == sent_idx],
            k=min(len(document) - 1, self.sents_nb),
        )

        return (
            [document[i] for i in selected_sents_idx if i < sent_idx],
            [document[i] for i in selected_sents_idx if i > sent_idx],
        )


class SameWordSelector(ContextSelector):
    """A context selector that randomly choose a sentence having a
    common name with the current sentence.

    """

    def __init__(self, sents_nb: int):
        self.sents_nb = sents_nb
        # nltk pos tagging dependency
        nltk.download("averaged_perceptron_tagger")

    def __call__(
        self, sent_idx: int, document: List[NERSentence]
    ) -> Tuple[List[NERSentence], List[NERSentence]]:
        """ """
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
            selected_sents_idx, k=min(self.sents_nb, len(selected_sents_idx))
        )

        return (
            [document[i] for i in selected_sents_idx if i < sent_idx],
            [document[i] for i in selected_sents_idx if i > sent_idx],
        )


class NeighborsContextSelector(ContextSelector):
    """A context selector that chooses nearby sentences."""

    def __init__(self, left_sents_nb: int, right_sents_nb: int) -> None:
        """
        :param left_sents_nb: number of left context sentences to select
        :param right_sents_nb: number of right context sentences to select
        """
        self.left_sents_nb = left_sents_nb
        self.right_sents_nb = right_sents_nb

    def __call__(
        self, sent_idx: int, document: List[NERSentence]
    ) -> Tuple[List[NERSentence], List[NERSentence]]:
        """ """
        return (
            document[max(0, sent_idx - self.left_sents_nb) : sent_idx],
            document[sent_idx + 1 : sent_idx + 1 + self.right_sents_nb],
        )


class NERDataset(Dataset):
    """
    :ivar documents: `List[List[NERSentence]]`
    :ivar tags: the set of all possible entity classes
    :ivar tags_nb: number of tags
    :ivar tags_to_id: `Dict[tag: str, id: int]`
    :ivar context_selectors:
    """

    def __init__(
        self,
        documents: List[List[NERSentence]],
        tags: Optional[Set[str]] = None,
        context_selectors: List[ContextSelector] = None,
    ) -> None:
        """
        :param documents:
        :param tags:
        :param context_selectors:
        """
        self.documents = documents

        if tags is None:
            self.tags = {
                tag for document in documents for sent in document for tag in sent.tags
            }
        else:
            self.tags = tags
        self.tags_nb = len(self.tags)
        self.tag_to_id: Dict[str, int] = {
            tag: i for i, tag in enumerate(sorted(list(self.tags)))
        }

        self.context_selectors = [] if context_selectors is None else context_selectors

        self.tokenizer: BertTokenizerFast = get_tokenizer()

    def tag_frequencies(self) -> Dict[str, float]:
        """
        :return: a mapping from token to its frequency
        """
        tags_count = defaultdict(int)
        for sent in self.sents():
            for tag in sent.tags:
                tags_count[tag] += 1
        total_count = sum(tags_count.values())
        return {tag: count / total_count for tag, count in tags_count.items()}

    def tag_weights(self) -> List[float]:
        """
        :return: a list of weights, ordered by `self.tags_to_id`.
            Each tag weight is computed as ``max_tags_frequency / tag_frequency``.
        """
        weights = [0.0] * len(self.tags)
        frequencies = self.tag_frequencies()
        max_frequency = max(frequencies.values())
        for tag, frequency in frequencies.items():
            weights[self.tag_to_id[tag]] = max_frequency / frequency
        return weights

    def sents(self) -> List[NERSentence]:
        """Return the list of sents of the datasets, ordered by documents."""
        return flattened(self.documents)

    def document_for_sent(self, sent_index: int) -> List[NERSentence]:
        """Get the document corresponding to the index of a sent."""
        counter = 0
        for document in self.documents:
            counter += len(document)
            if counter > sent_index:
                return document
        raise ValueError

    def __getitem__(self, index: int) -> Dict[str, List[str]]:
        """Get a BatchEncoding representing sentence at index, with its context

        .. note::

            As an addition to the classic huggingface BatchEncoding keys,
            a "tokens_labels_mask" is added to the outputed BatchEncoding.
            This masks denotes the difference between a sentence context
            (previous and next context) and the sentence itself. when
            concatenating a sentence and its context sentence, we obtain :

            ``[l1, l2, l3, ...] + [s1, s2, s3, ...] + [r1, r2, r3, ...]``

            with li being a token of the left context, si a token of the
            sentence and ri a token of the right context. The
            "tokens_labels_mask" is thus :

            ``[0, 0, 0, ...] + [1, 1, 1, ...] + [0, 0, 0, ...]``

            This mask is produced *before* tokenization by a huggingface
            tokenizer, and therefore corresponds to *tokens* and not to
            *wordpieces*.

        :param index:
        :return:
        """
        sents = self.sents()
        sent = sents[index]

        # retrieve context using context selectors
        document = self.document_for_sent(index)
        lcontexts = []
        rcontexts = []
        for selector in self.context_selectors:
            lcontext, rcontext = selector(document.index(sent), document)
            lcontexts += lcontext
            rcontexts += rcontext

        # add a dummy sentence with a separator if needed to inform
        # the model that sentences on the left and right are
        # contextuals
        if len(lcontexts) > 0:
            lcontexts = lcontexts + [NERSentence(["[SEP]"], ["O"])]
        if len(rcontexts) > 0:
            rcontexts = [NERSentence(["[SEP]"], ["O"])] + rcontexts

        # construct a new sentence with the retrieved context
        sent = NERSentence(sent.tokens, sent.tags, lcontexts, rcontexts)

        flattened_left_context = flattened([s.tokens for s in sent.left_context])
        flattened_right_context = flattened([s.tokens for s in sent.right_context])

        # create a BatchEncoding using huggingface tokenizer
        batch = self.tokenizer(
            flattened_left_context + sent.tokens + flattened_right_context,
            is_split_into_words=True,
        )  # type: ignore

        # create tokens_labels_mask
        batch["tokens_labels_mask"] = [0] * len(
            flattened([s.tags for s in sent.left_context])
        )
        batch["tokens_labels_mask"] += [1] * len(sent.tags)
        batch["tokens_labels_mask"] += [0] * len(
            flattened([s.tags for s in sent.right_context])
        )
        assert len([i for i in batch["tokens_labels_mask"] if i == 1]) == len(
            sents[index].tags
        )

        # manual truncation : this can deal with the case where we
        # need to truncate left
        truncation_direction = (
            "right"
            if len(flattened_left_context) < len(flattened_right_context)
            else "left"
        )
        batch = truncate_batch(batch, truncation_direction)

        # align tokens labels with wordpiece
        return align_tokens_labels_(
            batch,
            flattened([s.tags for s in sent.left_context])
            + sent.tags
            + flattened([s.tags for s in sent.right_context]),
            self.tag_to_id,
        )

    def __len__(self) -> int:
        return len(self.sents())
