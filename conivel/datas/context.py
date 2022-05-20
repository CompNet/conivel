from typing import List, Tuple
import random

import nltk

from conivel.datas import NERSentence


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
