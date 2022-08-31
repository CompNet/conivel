from typing import Dict, List, Literal, Optional, Set
from dataclasses import dataclass, field

import torch
from transformers import BertForTokenClassification  # type: ignore
from transformers.tokenization_utils_base import BatchEncoding

from conivel.datas import batch_to_device

from conivel.datas.dataset import NERDataset
from conivel.datas.dataset_utils import dataset_batchs


@dataclass
class PredictionOutput:

    #: a list of tags per sentence
    tags: List[List[str]] = field(default_factory=lambda: [])

    #: embedding of each tokens one tensor per sentence. When a token
    #: is composed of several wordpieces, its embedding is the mean of
    #: the embedding of the composing wordpieces. Each tensor is of
    #: shape ``(sentence_size, hidden_size)``. Only last layer
    #: embeddings are considered
    embeddings: Optional[List[torch.Tensor]] = None

    #: attention between tokens, one tensor per sentence. Each tensor
    #: is of shape ``(layers_nb, heads_nb, sentence_size, sentence_size)``
    #: . When a token is composed of several wordpieces
    attentions: Optional[List[torch.Tensor]] = None

    #: prediction scores, one tensor per sentence. Each tensor is : of
    # shape ``(sentence_size, vocab_size)`` (vocab size being the
    # number of different possible tags). When a token is composed of :
    # several wordpieces, its prediction score is the mean of the :
    # prediction scores.
    scores: Optional[List[torch.Tensor]] = None


def _get_batch_tags(
    batch: BatchEncoding, logits: torch.Tensor, id2label: Dict[int, str]
) -> List[List[str]]:
    """Get sequence of tags for a batch

    .. note::

        the ``words_labels_mask`` key is respected and
        context sentences tags are *not* extracted


    :param batch:
    :param logits: tensor of shape ``(batch_size, seq_size,
        vocab_size)``
    :param id2label:

    :return: a list of sentences tags
    """
    batch_size, seq_size = batch["input_ids"].shape  # type: ignore

    tags_indexs = torch.argmax(logits, dim=2)

    batch_tags = []

    for i in range(batch_size):

        token_to_word = [batch.token_to_word(i, token_index=j) for j in range(seq_size)]
        start_index = min(filter(lambda e: not e is None, token_to_word))
        token_to_word = [t - start_index if not t is None else t for t in token_to_word]
        tags_nb = sum(batch["words_labels_mask"][i])  # type: ignore

        sent_tags = ["O"] * tags_nb
        ignored_words: Set[int] = set()

        for j in range(seq_size):

            word_index = token_to_word[j]
            if word_index is None:
                continue

            if not batch["words_labels_mask"][i][word_index]:  # type: ignore
                ignored_words.add(word_index)
                continue

            tag_index = int(tags_indexs[i][j].item())
            sent_index = word_index - len(ignored_words)

            try:
                sent_tags[sent_index] = id2label[tag_index]
            except IndexError:
                breakpoint()

        batch_tags.append(sent_tags)

    return batch_tags


def _get_batch_embeddings(
    batch: BatchEncoding, embeddings: torch.Tensor
) -> List[torch.Tensor]:
    """Extract tokens embeddings from a batch

    .. note::

        the ``words_labels_mask`` key is respected
        and context sentences are *not* extracted

    :param batch:
    :param embeddings: a tensor of shape ``(batch_size, seq_size,
        hidden_size)``.  The embedding of each word is the mean of the
        embeddings of its subtokens.

    :return: a list of tensors of shape ``(sentence_size, hidden_size)``
             (one per batch sentence)
    """
    batch_size, seq_size = batch["input_ids"].shape  # type: ignore

    batch_embeddings: List[torch.Tensor] = []

    for i in range(batch_size):

        token_to_word = [batch.token_to_word(i, token_index=j) for j in range(seq_size)]
        start_index = min(filter(lambda e: not e is None, token_to_word))
        token_to_word = [t - start_index if not t is None else t for t in token_to_word]
        words_nb = sum(batch["words_labels_mask"][i])  # type: ignore

        sent_embeddings = [[] for _ in range(words_nb)]
        ignored_words = set()

        for j in range(seq_size):

            word_index = token_to_word[j]
            if word_index is None:
                continue

            if not batch["words_labels_mask"][i][word_index]:  # type: ignore
                ignored_words.add(word_index)
                continue

            sent_index = word_index - len(ignored_words)
            sent_embeddings[sent_index].append(embeddings[i][j])

        # reduce word embeddings to be the mean of the embeddings of
        # the subtokens composing them
        sent_embeddings = [torch.stack(embs, dim=0) for embs in sent_embeddings]
        sent_embeddings = [torch.mean(embs, dim=0) for embs in sent_embeddings]
        # (words_nb, hidden_size)
        sent_embeddings = torch.stack(sent_embeddings)
        batch_embeddings.append(sent_embeddings)

    return batch_embeddings


def _get_batch_scores(batch: BatchEncoding, logits: torch.Tensor) -> List[torch.Tensor]:
    """
    .. note::

        the ``words_labels_mask`` key is respected
        and context sentences are *not* extracted

    :param batch:
    :param logits: a tensor of shape ``(batch_size, seq_size,
        vocab_size)``

    :return: a list of tensors of shape ``(sentence_size,
             vocab_size)`` (one tensor per sentence).
    """
    batch_size, seq_size, vocab_size = logits.shape
    device = logits.device

    scores = torch.softmax(logits, dim=2)
    batch_scores = []

    for i in range(batch_size):

        token_to_word = [batch.token_to_word(i, token_index=j) for j in range(seq_size)]
        start_index = min(filter(lambda e: not e is None, token_to_word))
        token_to_word = [t - start_index if not t is None else t for t in token_to_word]
        words_nb = sum(batch["words_labels_mask"][i])  # type: ignore

        sent_scores = [[] for _ in range(words_nb)]
        ignored_words = set()

        for j in range(seq_size):

            word_index = token_to_word[j]
            if word_index is None:
                continue

            if not batch["words_labels_mask"][i][word_index]:  # type: ignore
                ignored_words.add(word_index)
                continue

            sent_index = word_index - len(ignored_words)
            sent_scores[sent_index].append(scores[i][j])

        # reduce word scores to be the mean of the scores of the
        # subtokens composing them. The result is a list of tensors of
        # shape (vocab_size).
        # note: its possible that a word has no score for its
        # subtokens. This can happen when the sentence was truncated :
        # at this point, there are no predictions for some words. In
        # those cases, we output scores of 0 for each class.
        sent_scores = [
            torch.mean(torch.stack(sc), dim=0)
            if len(sc) > 0
            else torch.zeros(vocab_size).to(device)  # no prediction for this word
            for sc in sent_scores
        ]
        # (sentence_size, vocab_size)
        sent_scores = torch.stack(sent_scores)
        batch_scores.append(sent_scores)

    return batch_scores


def _get_batch_attentions(
    batch: BatchEncoding, attentions: torch.Tensor
) -> List[torch.Tensor]:
    """Extract attention map from a batch

    .. note::

        the ``words_labels_mask`` key is *not* respected,
        and attentions from context sentences is therefore
        extracted. This is so attention with context
        sentences can be examinated.

    :param batch:
    :param attentions: a tensor of shape ``(layers_nb, heads_nb,
        batch_size, seq_size, seq_size)``

    :return: a list of tensors of shape ``(layers_nb, heads_nb,
             sentence+context_size, sentence+context_size)`` (one per
             sentence)
    """
    batch_size, seq_size = batch["input_ids"].shape  # type: ignore

    batch_attentions: List[torch.Tensor] = []

    for i in range(batch_size):

        token_to_word = [batch.token_to_word(i, token_index=j) for j in range(seq_size)]
        words_nb = len({k for k in token_to_word if not k is None})

        sent_attentions = [[[] for _ in range(words_nb)] for _ in range(words_nb)]

        for j in range(seq_size):

            word_index = token_to_word[j]
            if word_index is None:
                continue

            for k in range(seq_size):

                other_word_index = token_to_word[k]
                if other_word_index is None:
                    continue

                sent_attentions[word_index][other_word_index].append(
                    attentions[:, :, i, j, k]
                )

        # list(tokens) of list(tokens) of tensors of shape (layers_nb,
        # heads_nb)
        sent_attentions = [
            [
                torch.mean(torch.stack(wp_attentions), dim=0)
                for wp_attentions in word_attentions
            ]
            for word_attentions in sent_attentions
        ]
        # list of tensors of shape (words_nb, layers_nb, heads_nb)
        sent_attentions = [
            torch.stack(word_attentions) for word_attentions in sent_attentions
        ]
        # (words_nb, words_nb, layers_nb, heads_nb)
        sent_attentions = torch.stack(sent_attentions)
        # (layers_nb, heads_nb, words_nb, words_nb)
        sent_attentions = torch.permute(sent_attentions, (2, 3, 0, 1))
        batch_attentions.append(sent_attentions)

    return batch_attentions


def predict(
    model: BertForTokenClassification,
    dataset: NERDataset,
    batch_size: int = 4,
    quiet: bool = False,
    device_str: Literal["cuda", "cpu", "auto"] = "auto",
    additional_outputs: Optional[
        Set[Literal["embeddings", "scores", "attentions"]]
    ] = None,
    transfer_additional_outputs_to_cpu: bool = True,
) -> PredictionOutput:
    """perform prediction for a dataset

    :param model: a trained NER model
    :param dataset: a ``NERDataset``
    :param batch_size: batch size to use for prediction
    :param quiet: if ``True``, tqdm wont display a progress bar
    :param device_str: torch device to use for prediction

    :param additional_outputs: a set of possible additional outputs,
        between :

            - ``'embeddings'`` : a list of tensors of shape
              ``(sentence_size, hidden_size)``, one per sentence.

            - ``'scores'`` : a list of prediction score tensors of
              shape ``(sentence_size, vocab_size)``, one per sentence.

            - ``'attentions'`` : a list of attentions tensors of shape
              ``(layers_nb, heads_nb, sentence+context_size,
              sentence+context_size)``

    :param transfer_additional_outputs_to_cpu: when ``True``,
        transfers any additional output tensor to cpu.  This is useful
        to avoid reaching the GPU ram limit when, for example, making
        predictions on a sufficiently large dataset.
    """
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if additional_outputs is None:
        additional_outputs = set()

    model = model.eval()
    model = model.to(device)

    prediction = PredictionOutput(
        embeddings=[] if "embeddings" in additional_outputs else None,
        scores=[] if "scores" in additional_outputs else None,
        attentions=[] if "attentions" in additional_outputs else None,
    )

    with torch.no_grad():

        for batch in dataset_batchs(dataset, batch_size, quiet=quiet):

            local_batch_size, seq_size = batch["input_ids"].shape  # type: ignore
            lb, s = local_batch_size, seq_size

            batch = batch_to_device(batch, device)

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                output_attentions=True,
            )
            assert out.logits.shape == (lb, s, len(model.config.id2label))

            prediction.tags += _get_batch_tags(batch, out.logits, model.config.id2label)

            if "embeddings" in additional_outputs:
                embeddings = _get_batch_embeddings(batch, out.hidden_states[-1])
                if transfer_additional_outputs_to_cpu:
                    embeddings = [e.to(torch.device("cpu")) for e in embeddings]
                prediction.embeddings += embeddings  # type: ignore

            if "scores" in additional_outputs:
                scores = _get_batch_scores(batch, out.logits)  # type: ignore
                if transfer_additional_outputs_to_cpu:
                    scores = [s.to(torch.device("cpu")) for s in scores]
                prediction.scores += scores  # type: ignore

            if "attentions" in additional_outputs:
                attentions = _get_batch_attentions(
                    batch, torch.stack(out.attentions).transpose(1, 2)
                )  # type: ignore
                if transfer_additional_outputs_to_cpu:
                    attentions = [a.to(torch.device("cpu")) for a in attentions]
                prediction.attentions += attentions  # type: ignore

    return prediction
