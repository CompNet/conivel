from __future__ import annotations
import math, itertools, copy, random
from typing import TYPE_CHECKING, Set, List, Optional, Dict, Tuple, cast
from collections import defaultdict

from torch.utils.data import Dataset
from transformers import BertTokenizerFast  # type: ignore
from transformers.tokenization_utils_base import BatchEncoding

from conivel.datas import NERSentence, align_tokens_labels_
from conivel.utils import flattened, get_tokenizer

if TYPE_CHECKING:
    from conivel.datas.context import ContextSelector


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
        context_selectors: Optional[List[ContextSelector]] = None,
        tokenizer: Optional[BertTokenizerFast] = None,
    ) -> None:
        """
        :param documents:
        :param tags:
        :param context_selectors:
        """
        self.documents = documents

        if tags is None:
            self.tags = set()
            for document in documents:
                for sent in document:
                    self.tags = self.tags.union(sent.tags_set())
        else:
            self.tags = tags
        self.tags.add("O")
        self.tags_nb = len(self.tags)
        self.tag_to_id: Dict[str, int] = {
            tag: i for i, tag in enumerate(sorted(list(self.tags)))
        }
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}

        self.context_selectors = [] if context_selectors is None else context_selectors

        if tokenizer is None:
            tokenizer = get_tokenizer()
        self.tokenizer = cast(BertTokenizerFast, tokenizer)

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

    def kfolds(
        self, k: int, shuffle: bool = False, shuffle_seed: Optional[int] = None
    ) -> List[Tuple[NERDataset, NERDataset]]:
        """Return a kfold of the current dataset

        :param k: number of folds
        :param shuffle: if ``True``, documents are shuffled before
            splitting
        :param shuffle_seed: the seed to use when shuffling.  If
            ``None``, the global random generator is used.

        :return: a list of ``k`` tuples, each tuple being of the form
                 ``(train_set, test_set)``.
        """
        documents = copy.copy(self.documents)
        if shuffle:
            if shuffle_seed is None:
                random.shuffle(documents)
            else:
                random.Random(shuffle_seed).shuffle(documents)

        fold_size = math.ceil(len(documents) / k)
        folds = []
        for i in range(k):
            test_start = fold_size * i
            test_end = fold_size * (i + 1)
            test_set = documents[test_start:test_end]
            train_set = documents[:test_start] + documents[test_end:]
            folds.append((train_set, test_set))

        return [
            (
                NERDataset(train, self.tags, self.context_selectors, self.tokenizer),
                NERDataset(test, self.tags, self.context_selectors, self.tokenizer),
            )
            for train, test in folds
        ]

    @staticmethod
    def concatenated(datasets: List[NERDataset]) -> NERDataset:
        """Concatenate several datasets into a single one

        .. note::

            all datasets should have the same context selectors

        :param datasets: list of datasets to concatenate together
        """
        assert len(datasets) >= 2

        # check that all datasets have same context selectors
        for d1, d2 in itertools.combinations(datasets, 2):
            assert d1.context_selectors == d2.context_selectors

        # try to "smartly" select a tokenizer by taking the first
        # tokenizer from ``datasets`` that is not ``None``
        tokenizer = None
        for dataset in datasets:
            if not dataset.tokenizer is None:
                tokenizer = dataset.tokenizer
                break

        return NERDataset(
            flattened([dataset.documents for dataset in datasets]), tokenizer=tokenizer
        )

    def document_for_sent(self, sent_index: int) -> List[NERSentence]:
        """Get the document corresponding to the index of a sent."""
        counter = 0
        for document in self.documents:
            counter += len(document)
            if counter > sent_index:
                return document
        raise ValueError

    def sent_document_index(self, sent_index: int) -> int:
        """Get the index of a sent in its document

        :param sent_index: the global index of the sent in the dataset
        :return: the index of the given sent in its document
        """
        index_in_doc = sent_index
        for document in self.documents:
            if index_in_doc < len(document):
                return index_in_doc
            index_in_doc -= len(document)
        raise ValueError

    def __getitem__(self, index: int) -> BatchEncoding:
        """Get a BatchEncoding representing sentence at index, with
        its context

        .. note::

            As an addition to the classic huggingface BatchEncoding keys,
            a ``"words_labels_mask"`` is added to the outputed BatchEncoding.
            This masks denotes the difference between a sentence context
            (previous and next context) and the sentence itself. when
            concatenating a sentence and its context sentence, we obtain :

            ``[l1, l2, l3, ...] + [s1, s2, s3, ...] + [r1, r2, r3, ...]``

            with li being a token of the left context, si a token of the
            sentence and ri a token of the right context. The
            ``"words_labels_mask"`` is thus :

            ``[0, 0, 0, ...] + [1, 1, 1, ...] + [0, 0, 0, ...]``

            This mask is produced *after* tokenization by a huggingface
            tokenizer, and therefore corresponds to *wordpieces* and not to
            *original tokens*.

        .. note::

            if ``len(self.context_selectors) == 0``, sentences left and right
            contexts are used as context.

        :param index:

        :return:
        """
        sents = self.sents()
        sent = sents[index]

        # retrieve context using context selectors
        if len(self.context_selectors) > 0:
            document = self.document_for_sent(index)
            lcontexts = []
            rcontexts = []
            for selector in self.context_selectors:
                lcontext, rcontext = selector(
                    self.sent_document_index(index), tuple(document)
                )
                lcontexts += lcontext
                rcontexts += rcontext
        else:
            lcontexts = sent.left_context
            rcontexts = sent.right_context

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
        truncation_side = (
            "right"
            if len(flattened_left_context) <= len(flattened_right_context)
            else "left"
        )
        self.tokenizer.truncation_side = truncation_side
        batch = self.tokenizer(
            ["[CLS]"]
            + flattened_left_context
            + sent.tokens
            + flattened_right_context
            + ["[SEP]"],
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,  # no hidden magic please
        )  # type: ignore

        labels = (
            # [CLS]
            ["O"]
            + flattened([s.tags for s in sent.left_context])
            + sent.tags
            + flattened([s.tags for s in sent.right_context])
            # [SEP]
            + ["O"]
        )

        # align tokens labels with wordpiece
        batch = align_tokens_labels_(batch, labels, self.tag_to_id)

        # [CLS]
        words_labels_mask = (
            # [CLS]
            [0]
            # left context
            + [0] * len(flattened([s.tags for s in sent.left_context]))
            # sentence
            + [1] * len(sent.tags)
            # right context
            + [0] * len(flattened([s.tags for s in sent.right_context]))
            # [SEP]
            + [0]
        )
        batch["words_labels_mask"] = words_labels_mask

        return batch

    def __len__(self) -> int:
        return len(self.sents())
