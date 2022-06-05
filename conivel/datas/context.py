from typing import List, Optional, Tuple
import random

import nltk
from sacred.run import Run

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
import torch
from torch.utils.data import Dataset, DataLoader, dataloader
from transformers import BertForTokenClassification, BertForSequenceClassification, BertTokenizerFast, DataCollatorWithPadding  # type: ignore
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
    label: Optional[Literal[0, 1]]


class ContextSelectionDataset(Dataset):
    """
    :todo: watch out for context direction (does it come from the left
    or the right ?), it might affect performance and is unhandled as
    of now.  Maybe we should rethink the input method and enclose
    context in special tokens such as '[CTX]' ?
    """

    def __init__(self, examples: List[ContextSelectionExample]) -> None:
        self.examples = examples
        self.tokenizer: BertTokenizerFast = get_tokenizer()

    def __len__(self) -> int:
        return len(self.examples)

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

        if not example.label is None:
            batch["label"] = example.label

        return batch

    def to_jsonifiable(self) -> List[dict]:
        return [vars(example) for example in self.examples]


class NeuralContextSelector(ContextSelector):
    """A context selector powered by BERT"""

    def __init__(
        self,
        pretrained_model_name: str,
        heuristic_retrieval_sents_nb: int,
        batch_size: int,
    ) -> None:
        """
        :param batch_size: batch size used at inference
        """
        self.ctx_classifier: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(pretrained_model_name) # type: ignore  

        self.heuristic_retrieval_sents_nb = heuristic_retrieval_sents_nb
        self.same_word_selector = SameWordSelector(heuristic_retrieval_sents_nb)

        self.batch_size = batch_size

    def __call__(
        self, sent_idx: int, document: List[NERSentence]
    ) -> Tuple[List[NERSentence], List[NERSentence]]:
        """"""
        sent = document[sent_idx]

        # get self.heuristic_retrieval_sents_nb potentially important
        # context sentences
        left_ctx, right_ctx = self.heuristic_retrieve_ctx(sent_idx, document)
        ctx_sents = left_ctx + right_ctx

        # no context retrieved by heuristic : nothing to do
        if len(ctx_sents) == 0:
            return ([], [])

        # prepare datas for inference
        dataset = ContextSelectionDataset(
            [
                ContextSelectionExample(sent.tokens, ctx_sent.tokens, None)
                for ctx_sent in ctx_sents
            ]
        )
        data_collator = DataCollatorWithPadding(dataset.tokenizer)  # type: ignore
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator
        )

        # inference using self.ctx_classifier
        self.ctx_classifier = self.ctx_classifier.eval()
        scores = torch.zeros((0,))
        with torch.no_grad():
            for X in dataloader:
                # out.logits is of shape (batch_size, 2)
                out = self.ctx_classifier(
                    X["input_ids"],
                    token_type_ids=X["token_type_ids"],
                    attention_mask=X["attention_mask"],
                )
                scores = torch.cat([scores, out.logits[:, 1]], dim=0)

        # now scores should be of shape
        # (self.heuristic_retrieval_sents_nb). We keep the context
        # sentence with the max score.
        best_ctx_idx: int = torch.argmax(scores, dim=0).item()  # type: ignore

        # best_ctx_idx is from left context
        if best_ctx_idx < len(left_ctx):
            return ([ctx_sents[best_ctx_idx]], [])
        # best_ctx_idx is from right context
        return ([], [ctx_sents[best_ctx_idx]])

    def heuristic_retrieve_ctx(
        self, sent_idx: int, document: List[NERSentence]
    ) -> Tuple[List[NERSentence], List[NERSentence]]:
        """Retrieve potentially useful context sentences to help
        predict sent at index ``sent_idx``.

        :param sent_idx: index of the sent for which NER predictions
            are made.
        :param document: the predicted sentence's document.

        :return: ``(left_context, right_context)``
        """
        return self.same_word_selector(sent_idx, document)

    @staticmethod
    def generate_context_dataset(
        ner_model: BertForTokenClassification,
        train_dataset: NERDataset,
        batch_size: int,
        samples_per_sent: int,
        max_examples_nb: Optional[int] = None,
        _run: Optional[Run] = None,
    ) -> ContextSelectionDataset:
        """Generate a context selection training dataset.

        The process is as follows :

            1. Make predictions for a NER dataset using an already
               trained NER model.

            2. For all erroneous sentence predictions, sample a bunch
               of possible context sentences using some heuristic, and
               retry predictions with those context sentences.
               Context sentences not able to fix wrong predictions are
               negative example of context retrieval, while context
               sentences that can fix wrong predictions are positive
               examples.

        .. note::

            For now, uses ``SameWordSelector`` as sampling heuristic.

        :todo: make a choice on heuristic

        :param ner_model: an already trained NER model used to
            generate initial predictions
        :param train_dataset: NER dataset used to extract examples
        :param batch_size: batch size used for NER inference
        :param samples_per_sent: number of context selection samples
            to generate per wrongly predicted sentence
        :param max_examples_nb: max number of examples in the
            generated dataset.  If ``None``, no limit is applied.
        :param _run: The current sacred run.  If not ``None``, will be
            used to record generation metrics.

        :return: a ``ContextSelectionDataset`` that can be used to
                 train a context selector.
        """

        predictions = cast(
            List[List[str]], predict(ner_model, train_dataset, batch_size=batch_size)
        )

        preliminary_ctx_selector = SameWordSelector(samples_per_sent)

        ctx_selection_examples = []
        for sent_i, (sent, sent_prediction) in tqdm(
            enumerate(zip(train_dataset.sents(), predictions)), total=len(predictions)
        ):
            # prediction from the NER model was correct : nothing to do
            if sent.tags == sent_prediction:
                continue
            document = train_dataset.document_for_sent(sent_i)

            if (
                not max_examples_nb is None
                and len(ctx_selection_examples) >= max_examples_nb
            ):
                ctx_selection_examples = ctx_selection_examples[:max_examples_nb]
                break

            # retrieve n context sentences
            index_in_doc = train_dataset.sent_document_index(sent_i)
            left_ctx_sents, right_ctx_sents = preliminary_ctx_selector(
                index_in_doc, document
            )
            sent_and_ctx = [
                NERSentence(sent.tokens, sent.tags, left_context=[ctx_sent])
                for ctx_sent in left_ctx_sents
            ]
            sent_and_ctx += [
                NERSentence(sent.tokens, sent.tags, right_context=[ctx_sent])
                for ctx_sent in right_ctx_sents
            ]

            # generate positive and negative examples
            # TODO: positive / negative ratio
            preds_with_ctx = predict(
                ner_model,
                NERDataset([sent_and_ctx], train_dataset.tags),
                quiet=True,
                batch_size=batch_size,
            )
            preds_with_ctx = cast(List[List[str]], preds_with_ctx)
            for pred_with_ctx, ctx_sent in zip(
                preds_with_ctx, left_ctx_sents + right_ctx_sents
            ):
                ctx_was_helpful = pred_with_ctx == sent.tags
                example = ContextSelectionExample(
                    sent.tokens, ctx_sent.tokens, 1 if ctx_was_helpful else 0
                )
                ctx_selection_examples.append(example)

        if not _run is None:
            _run.log_scalar(
                "context_dataset_generation.examples_nb", len(ctx_selection_examples)
            )

            pos_examples_nb = sum([ex.label for ex in ctx_selection_examples])
            neg_examples_nb = len(ctx_selection_examples) - pos_examples_nb
            try:
                _run.log_scalar(
                    "context_dataset_generation.pos_neg_ratio",
                    pos_examples_nb / neg_examples_nb,
                )
            except ZeroDivisionError:
                pass

        return ContextSelectionDataset(ctx_selection_examples)

    @staticmethod
    def train_context_selector(
        ctx_dataset: ContextSelectionDataset,
        epochs_nb: int,
        batch_size: int,
        learning_rate: float,
        _run: Optional[Run] = None,
    ) -> BertForSequenceClassification:
        """Instantiate and train a context classifier.

        :param ner_model: an already trained NER model used to
            generate the context selection dataset.
        :param train_dataset: NER dataset used to generate the context
            selection dataset.
        :param epochs_nb: number of training epochs.
        :param batch_size:
        :param _run: current sacred run.  If not ``None``, will be
            used to record training metrics.

        :return: a trained ``BertForSequenceClassification``
        """
        ctx_classifier = BertForSequenceClassification.from_pretrained(
            "bert-base-cased"
        )  # type: ignore

        optimizer = torch.optim.AdamW(ctx_classifier.parameters(), lr=learning_rate)

        data_collator = DataCollatorWithPadding(ctx_dataset.tokenizer)  # type: ignore
        dataloader = DataLoader(
            ctx_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
        )

        for _ in range(epochs_nb):

            epoch_losses = []
            ctx_classifier = ctx_classifier.train()

            data_tqdm = tqdm(dataloader)
            for X in data_tqdm:

                optimizer.zero_grad()

                out = ctx_classifier(
                    X["input_ids"],
                    token_type_ids=X["token_type_ids"],
                    attention_mask=X["attention_mask"],
                    labels=X["labels"],
                )
                out.loss.backward()
                optimizer.step()

                if not _run is None:
                    _run.log_scalar("training.loss", out.loss.item())

                data_tqdm.set_description(f"loss : {out.loss.item():.3f}")
                epoch_losses.append(out.loss.item())

            mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            tqdm.write(f"epoch mean loss : {mean_epoch_loss:.3f}")
            if not _run is None:
                _run.log_scalar("training.mean_epoch_loss", mean_epoch_loss)

        return ctx_classifier
