from typing import List, Optional, Union, Tuple, cast
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForTokenClassification

from conivel.datas import (
    batch_to_device,
    DataCollatorForTokenClassificationWithBatchEncoding,
)
from conivel.datas.dataset import NERDataset


@dataclass
class PredictionOutput:

    #: a list of tags per sentence
    tags: List[List[str]] = field(default_factory=lambda: [])

    #: embedding of each tokens one tensor per sentence. When a token
    #: is composed of several wordpieces, its embedding is the mean of
    #: the embedding of the composing wordpieces. Each tensor is of
    #: shape ``(sentence_size, hidden_size)``.
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


def predict(
    model: BertForTokenClassification,
    dataset: NERDataset,
    batch_size: int = 4,
    quiet: bool = False,
    return_embeddings: bool = False,
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
