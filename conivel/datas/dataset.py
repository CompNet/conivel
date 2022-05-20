from typing import Set, List, Optional, Dict
from collections import defaultdict

from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from conivel.datas import NERSentence, align_tokens_labels_
from conivel.datas.context import ContextSelector
from conivel.utils import flattened, get_tokenizer


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

    def __getitem__(self, index: int) -> BatchEncoding:
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
        truncation_side = (
            "right"
            if len(flattened_left_context) < len(flattened_right_context)
            else "left"
        )
        self.tokenizer.truncation_side = truncation_side
        batch = self.tokenizer(
            flattened_left_context + sent.tokens + flattened_right_context,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
        )  # type: ignore

        # create tokens_labels_mask
        batch["tokens_labels_mask"] = [0] * len(
            flattened([s.tags for s in sent.left_context])
        )
        batch["tokens_labels_mask"] += [1] * len(sent.tags)
        batch["tokens_labels_mask"] += [0] * len(
            flattened([s.tags for s in sent.right_context])
        )

        # align tokens labels with wordpiece
        batch = align_tokens_labels_(
            batch,
            flattened([s.tags for s in sent.left_context])
            + sent.tags
            + flattened([s.tags for s in sent.right_context]),
            self.tag_to_id,
        )

        return batch

    def __len__(self) -> int:
        return len(self.sents())