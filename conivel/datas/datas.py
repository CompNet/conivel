from __future__ import annotations
from typing import List, Dict, Literal, Set, Union, Optional
from dataclasses import dataclass, field

from itertools import chain
from more_itertools import windowed
import torch
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


@dataclass(frozen=True)
class NERSentence:
    tokens: List[str] = field(default_factory=lambda: [])
    tags: List[str] = field(default_factory=lambda: [])
    left_context: List[NERSentence] = field(default_factory=lambda: [])
    right_context: List[NERSentence] = field(default_factory=lambda: [])

    def __len__(self) -> int:
        assert len(self.tokens) == len(self.tags)
        return len(self.tokens)

    def len_with_ctx(self) -> int:
        out_len = len(self)
        for sent in self.left_context + self.right_context:
            out_len += len(sent)
        return out_len

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

    def __hash__(self) -> int:
        assert len(self.tokens) == len(self.tags)
        return hash(
            tuple(self.tokens)
            + tuple(self.tags)
            + ("l",)
            + tuple(self.left_context)
            + ("r",)
            + tuple(self.right_context)
        )

    def tags_set(self) -> Set[str]:
        tags = set(self.tags)
        for sent in self.left_context + self.right_context:
            tags = tags.union(sent.tags_set())
        return tags

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
    batch_encoding: BatchEncoding,
    labels: List[str],
    all_labels: Dict[str, int],
) -> BatchEncoding:
    """Modify a huggingface single batch encoding by adding tokens
    labels, taking wordpiece into account

    .. note::

        Adapted from https://huggingface.co/docs/transformers/custom_datasets#tok_ner

    :param batch_encoding: ``'labels'`` key will be added to the batch
        encoding.  It must be a single batch.
    :param labels: list of per-word labels.  ``None`` labels will be
        given -100 label, in order to be ignored by torch loss
        functions.
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
    """Same as ``transformers.DataCollatorForTokenClassification``,
    except it :

        - correctly returns a ``BatchEncoding`` object with correct
          ``encodings`` attribute.

        - wont try to convert the key ``'words_labels_mask'`` that is
          used to tell apart which tokens are to be predicted vs
          context

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

        # ignore "words_labels_mask"
        return BatchEncoding(
            {
                k: torch.tensor(v, dtype=torch.int64)
                if not k == "words_labels_mask"
                else v
                for k, v in batch.items()
            },
            encoding=batch.encodings,
        )
