from typing import List, Tuple
import random

import nltk

from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset


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


from typing import Literal, cast
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import BertForTokenClassification, BertForSequenceClassification  # type: ignore
from transformers import BertTokenizerFast  # type: ignore
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from conivel.utils import get_tokenizer
from conivel.predict import predict


@dataclass
class ContextSelectionExample:
    """A context selection example, to be used for training a context selector."""

    #: sentence on which NER is performed
    sent: List[str]
    #: context to assist during prediction
    context: List[str]
    #: 0 => context was not useful, 1 => context was useful
    label: Literal[0, 1]


class ContextSelectionDataset(Dataset):
    """"""

    def __init__(self, examples: List[ContextSelectionExample]) -> None:
        self.examples = examples
        self.tokenizer: BertTokenizerFast = get_tokenizer()

    def __getitem__(self, index: int) -> BatchEncoding:
        """Get a BatchEncoding representing example at index.

        :param index: index of the example to retrieve

        :return: a ``BatchEncoding``, with key ``'label'`` set.
        """
        example = self.examples[index]

        batch: BatchEncoding = self.tokenizer(
            example.sent + ["[SEP]"] + example.context,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
        )

        batch["label"] = example.label

        return batch


class NeuralContextSelector(ContextSelector):
    """A context selector powered by BERT"""

    def __init__(self, pretrained_model_name: str) -> None:
        self.ctx_classifier = BertForSequenceClassification.from_pretrained(
            pretrained_model_name
        )  # type: ignore

    def __call__(
        self, sent_idx: int, document: List[NERSentence]
    ) -> Tuple[List[NERSentence], List[NERSentence]]:
        """"""
        raise NotImplementedError

    @staticmethod
    def train_context_classifier(
        ner_model: BertForTokenClassification, train_dataset: NERDataset, epochs_nb: int
    ) -> BertForSequenceClassification:
        """Instantiate and train a context classifier.

        The principle is as follows :

            1. Generate a context selection training dataset.  This is
               done by :

                1. Making predictions for a NER dataset using an
                   already trained NER model.

                2. For all erroneous sentence predictions, sample a
                   bunch of possible context sentences using some
                   heuristic, and retry predictions with those context
                   sentences.  Context sentences not able to fix wrong
                   predictions are negative example of context
                   retrieval, while context sentences that can fix
                   wrong predictions are positive examples.

            2. Train a ``BertForSequenceClassification`` using the
               generated context selection training dataset

        :param ner_model: an already trained NER model used to
            generate the context selection dataset.
        :param train_dataset: NER dataset used to generate the context
            selection dataset.
        :param epochs_nb: number of training epochs.

        :return: a trained ``BertForSequenceClassification``
        """
        # Step 1 : generate a context selection dataset

        ## Step 1.1 : make predictions using the trained ner model
        # TODO: batch_size ?
        predictions = cast(List[List[str]], predict(ner_model, train_dataset))

        ## Step 2.2
        ctx_selection_examples = []
        for sent_i, (sent, sent_prediction) in enumerate(
            zip(train_dataset.sents(), predictions)
        ):
            # prediction was correct : nothing to do
            if sent.tags == sent_prediction:
                continue
            document = train_dataset.document_for_sent(sent_i)

        # Step 2 : create a context selector and train it using the
        # generated dataset
        ctx_classifier = BertForSequenceClassification.from_pretrained(
            "bert-base-cased"
        )  # type: ignore

        for epoch in range(epochs_nb):
            ctx_classifier = ctx_classifier.train()

            # TODO: train ctx_classifier

        return ctx_classifier
