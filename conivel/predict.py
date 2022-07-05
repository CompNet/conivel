from typing import Dict, List, Literal, Optional, Union, Tuple, cast
from statistics import mean
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForTokenClassification
from transformers import tokenization_utils_base
from transformers.tokenization_utils_base import BatchEncoding

from conivel.datas import (
    batch_to_device,
    DataCollatorForTokenClassificationWithBatchEncoding,
)
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
    embeddings: List[torch.Tensor] = field(default_factory=lambda: [])

    #: attention between tokens, one tensor per sentence. Each tensor
    #: is of shape ``(layers_nb, heads_nb, sentence_size, sentence_size)``
    #: . When a token is composed of several wordpieces
    attentions: List[torch.Tensor] = field(default_factory=lambda: [])

    #: prediction scores, one tensor per sentence. Each tensor is
    #: of shape ``(sentence_size)``. When a token is composed of
    #: several wordpieces, its prediction score is the mean of the
    #: prediction scores.
    scores: List[torch.Tensor] = field(default_factory=lambda: [])


def _get_batch_tags(
    batch: BatchEncoding, logits: torch.Tensor, id2label: Dict[int, str]
) -> List[List[str]]:
    """Get sequence of tags for a batch

    .. note::

        the ``tokens_labels_mask`` key is respected and
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
        tags_nb = len([k for k in token_to_word if not k is None])

        batch_tags.append(["O"] * tags_nb)

        for j in range(seq_size):

            if not batch["tokens_labels_mask"][i][j].item():  # type: ignore
                continue

            word_index = token_to_word[j]
            if word_index is None:
                continue

            tag_index = int(tags_indexs[i][j].item())
            batch_tags[-1][word_index] = id2label[tag_index]

    return batch_tags


def _get_batch_embeddings(
    batch: BatchEncoding, embeddings: torch.Tensor
) -> List[torch.Tensor]:
    """Extract tokens embeddings from a batch

    .. note::

        the ``tokens_labels_mask`` key is respected
        and context sentences are *not* extracted

    :param batch:
    :param embeddings: a tensor of shape ``(batch_size, seq_size,
        hidden_size)``.  The embedding of each word is the mean of the
        embeddings of its subtokens.

    :return: a list of tensors of shape ``(sentence_size, hidden_size)``
             (one per batch sentence)
    """
    batch_size, seq_size = batch["input_ids"].shape  # type: ignore

    batch_embeddings = []

    for i in range(batch_size):

        token_to_word = [batch.token_to_word(i, token_index=j) for j in range(seq_size)]
        words_nb = len([k for k in token_to_word if not k is None])

        sent_embeddings = [[] for _ in range(words_nb)]

        for j in range(seq_size):

            if not batch["tokens_labels_mask"][i][j].item():  # type: ignore
                continue

            word_index = token_to_word[j]
            if word_index is None:
                continue

            sent_embeddings[word_index].append(embeddings[i][j])

        # reduce word embeddings to be the mean of the embeddings of
        # the subtokens composing them
        sent_embeddings = [torch.stack(embs, dim=0) for embs in sent_embeddings]
        sent_embeddings = [torch.mean(embs, dim=0) for embs in sent_embeddings]
        batch_embeddings.append(sent_embeddings)

    return batch_embeddings


def _get_batch_scores(batch: BatchEncoding, logits: torch.Tensor) -> List[torch.Tensor]:
    """

    .. note::

        the ``tokens_labels_mask`` key is respected
        and context sentences are *not* extracted

    :param batch:
    :param logits: a tensor of shape ``(batch_size, seq_size, vocab_size)``

    :return: a list of tensors of shape ``(sentence_size)`` (one
             tensor per sentence).
    """
    batch_size, seq_size = batch["input_ids"].shape  # type: ignore

    scores = torch.softmax(logits, dim=2)
    scores = cast(torch.Tensor, torch.max(scores, dim=2))
    assert scores.shape == (batch_size, seq_size)

    batch_scores = []

    for i in range(batch_size):

        token_to_word = [batch.token_to_word(i, token_index=j) for j in range(seq_size)]
        words_nb = len([k for k in token_to_word if not k is None])

        sent_scores = [[] for _ in range(words_nb)]

        for j in range(seq_size):

            if not batch["tokens_labels_mask"][i][j].item():  # type: ignore
                continue

            word_index = token_to_word[j]
            if word_index is None:
                continue

            sent_scores[word_index].append(scores[i][word_index].item())

        # reduce word scores to be the mean of the scores of the
        # subtokens composing them
        sent_scores = torch.tensor([mean(sc) for sc in sent_scores])
        batch_scores.append(sent_scores)

    return batch_scores


def _get_batch_attentions(
    batch: BatchEncoding, attentions: torch.Tensor
) -> List[torch.Tensor]:
    """Extract attention map from a batch

    .. note::

        the ``tokens_labels_mask`` key is *not* respected,
        and attentions from context sentences is therefore
        extracted. This is so attention with context
        sentences can be examinated.

    :param batch:
    :param attentions: a tensor of shape ``(batch_size, layers_nb,
        heads_nb, seq_size, seq_size)``

    :return: a list of tensors of shape ``(layers_nb, heads_nb,
             sentence+context_size, sentence+context_size)`` (one per
             sentence)
    """
    batch_size, seq_size = batch["input_ids"].shape  # type: ignore

    batch_attentions = []

    for i in range(batch_size):

        token_to_word = [batch.token_to_word(i, token_index=j) for j in range(seq_size)]
        words_nb = len([k for k in token_to_word if not k is None])

        sent_attentions = [[[] for _ in range(words_nb)] for _ in range(words_nb)]

        for j in range(seq_size):

            if not batch["tokens_labels_mask"][i][j].item():  # type: ignore
                continue

            word_index = token_to_word[j]
            if word_index is None:
                continue

            for k in range(seq_size):

                if not batch["tokens_labels_mask"][i][k].item():  # type: ignore
                    continue

                other_word_index = token_to_word[k]
                if other_word_index is None:
                    continue

                sent_attentions[word_index][other_word_index].append(
                    attentions[:, :, i, j, k]
                )

        sent_attentions = [
            [
                torch.mean(torch.stack(wp_attentions))
                for wp_attentions in word_attentions
            ]
            for word_attentions in sent_attentions
        ]
        sent_attentions = torch.tensor(sent_attentions)
        batch_attentions.append(sent_attentions)

    return batch_attentions


def predict(
    model: BertForTokenClassification,
    dataset: NERDataset,
    batch_size: int = 4,
    quiet: bool = False,
    device_str: Literal["cuda", "cpu", "auto"] = "auto",
) -> PredictionOutput:
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model = model.eval()
    model = model.to(device)

    prediction = PredictionOutput()

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
            prediction.embeddings += _get_batch_embeddings(batch, out.hidden_states[-1])
            prediction.scores += _get_batch_scores(batch, out.logits)
            prediction.attentions += _get_batch_attentions(
                batch, torch.tensor(out.attentions)
            )

    return prediction


def predict_old(
    model: BertForTokenClassification,
    dataset: NERDataset,
    batch_size: int = 4,
    quiet: bool = False,
) -> PredictionOutput:
    """Predict NER labels for a dataset

    :todo: optim

    :param model: the model to use for prediction
    :param dataset: the dataset on which to do predictions
    :param batch_size:
    :param quiet:

    :return: a ``PredictionOutput`` object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layers_nb = model.config.num_hidden_layers
    heads_nb = model.config.num_attention_heads

    model = model.eval()
    model = model.to(device)

    data_collator = DataCollatorForTokenClassificationWithBatchEncoding(
        dataset.tokenizer  # type: ignore
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    predict_out = PredictionOutput()

    with torch.no_grad():

        for data in tqdm(dataloader, disable=quiet):

            local_batch_size = data["input_ids"].shape[0]
            seq_size = data["input_ids"].shape[1]

            data = batch_to_device(data, device)

            out = model(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                output_hidden_states=True,
                output_attentions=True,
            )
            assert out.logits.shape == (
                local_batch_size,
                seq_size,
                len(model.config.id2label),
            )

            tags_indexs = torch.argmax(out.logits, dim=2)
            assert tags_indexs.shape == (local_batch_size, seq_size)

            # (layers_nb, heads_nb, batch_size, sentence_size, sentence_size)
            batch_attentions = torch.stack(out.attentions)

            # for each sentence of a batch
            for i in range(local_batch_size):

                # tokens_labels_mask doesn't take into account special tokens such
                # as [CLS] : this allows us to ignore them
                prefix_tokens_nb = 0
                for token_label_mask in data["tokens_labels_mask"][i]:  # type: ignore
                    if token_label_mask == 1:
                        break
                    prefix_tokens_nb += 1

                tags_nb = len([k for k in data["tokens_labels_mask"][i] if k == 1])  # type: ignore
                predict_out.tags.append(["O"] * tags_nb)

                tokens_embeddings = [None] * tags_nb
                # (tags_nb, tags_nb)
                attentions = [[None] * tags_nb] * tags_nb

                # for each token
                for j in range(seq_size):

                    t_j = data.token_to_word(i, token_index=j)

                    # special token not corresponding to any word
                    if t_j is None:
                        continue

                    # word excluded by tokens_labels_mask
                    if data["tokens_labels_mask"][i][t_j] == 0:  # type: ignore
                        continue

                    word_index = t_j - prefix_tokens_nb
                    tag_index = tags_indexs[i][j].item()

                    predict_out.tags[-1][word_index] = model.config.id2label[tag_index]  # type: ignore

                    # embeddings
                    if tokens_embeddings[word_index] is None:
                        # (hidden_size)
                        last_hidden_state = out.hidden_states[-1][i, j, :]
                        tokens_embeddings[word_index] = last_hidden_state
                        # (1, hidden_size)
                        tokens_embeddings[word_index].unsqueeze_(0)  # type: ignore
                    else:
                        tokens_embeddings[word_index] = torch.cat(
                            [
                                # (n, hidden_size)
                                tokens_embeddings[word_index],
                                # (1, hidden_size)
                                out.hidden_states[-1][i, j, :].unsqueeze(0),
                            ],  # type: ignore
                            dim=0,
                        )

                    # attention
                    for k in range(tags_nb):
                        if attentions[word_index][k] is None:
                            for l in range(seq_size):
                                pass
                            # (layers_nb, heads_nb, sentence_size)
                            attentions[word_index][k] = out.attentions[:, :, i, j, :]
                        else:
                            pass

                # add sentence embedding to predict_out
                tokens_embeddings = cast(List[torch.Tensor], tokens_embeddings)  # type: ignore
                tokens_embeddings = [torch.mean(t, dim=0) for t in tokens_embeddings]
                predict_out.embeddings.append(torch.stack(tokens_embeddings, dim=0))

                # TODO: attention

    return predict_out
