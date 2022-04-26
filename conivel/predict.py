from typing import List, Optional, Union, Tuple, cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForTokenClassification

from conivel.datas import (
    batch_to_device,
    NERDataset,
    DataCollatorForTokenClassificationWithBatchEncoding,
)


def predict(
    model: BertForTokenClassification,
    dataset: NERDataset,
    batch_size: int = 4,
    quiet: bool = False,
    return_embeddings: bool = False,
) -> Union[List[List[str]], Tuple[List[List[str]], List[torch.Tensor]]]:
    """Predict NER labels for a dataset

    :param model: the model to use for prediction
    :param dataset: the dataset on which to do predictions
    :param batch_size:
    :param quiet:
    :param return_embeddings: if ``True``, also returns tokens embeddings.

    :return: a list of tags per sentence. If ``return_embeddings`` is
        ``True``, additionaly returns the final embeddings of each
        token in a tuple. Embeddings are returned as a list of torch
        tensor of shape ``(sentence_size, hidden_size)`` (one per sentence).
        When a token is composed of several wordpiece, its embedding is the
        mean of the embedding of the composing wordpieces.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.eval()
    model = model.to(device)

    data_collator = DataCollatorForTokenClassificationWithBatchEncoding(
        dataset.tokenizer
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    sent_tags: List[List[str]] = []
    sent_embeddings: List[torch.Tensor] = []

    with torch.no_grad():

        for data in tqdm(dataloader, disable=quiet):

            local_batch_size = data["input_ids"].shape[0]
            seq_size = data["input_ids"].shape[1]

            data = batch_to_device(data, device)

            out = model(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                output_hidden_states=return_embeddings,
            )
            assert out.logits.shape == (
                local_batch_size,
                seq_size,
                len(model.config.id2label),
            )

            tags_indexs = torch.argmax(out.logits, dim=2)
            assert tags_indexs.shape == (local_batch_size, seq_size)

            for i in range(local_batch_size):

                # tokens_labels_mask doesn't take into account special tokens such
                # as [CLS] : this allows us to ignore them
                prefix_tokens_nb = 0
                for token_label_mask in data["tokens_labels_mask"][i]:  # type: ignore
                    if token_label_mask == 1:
                        break
                    prefix_tokens_nb += 1

                tags_nb = len([k for k in data["tokens_labels_mask"][i] if k == 1])  # type: ignore
                sent_tags.append(["O"] * tags_nb)

                if return_embeddings:
                    tokens_embeddings = [None] * tags_nb

                for j in range(seq_size):

                    t_j = data.token_to_word(i, token_index=j)

                    if t_j is None:
                        continue

                    if data["tokens_labels_mask"][i][t_j] == 0:  # type: ignore
                        continue

                    word_index = t_j - prefix_tokens_nb
                    tag_index = tags_indexs[i][j].item()

                    sent_tags[-1][word_index] = model.config.id2label[tag_index]

                    if return_embeddings:
                        tokens_embeddings = cast(
                            List[Optional[torch.Tensor]], tokens_embeddings  # type: ignore
                        )
                        if tokens_embeddings[word_index] is None:
                            # (hidden_size)
                            last_hidden_state = out.hidden_states[-1][i, j, :]
                            tokens_embeddings[word_index] = last_hidden_state
                            # (1, hidden_size)
                            tokens_embeddings[word_index].unsqueeze_(0)
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

                if return_embeddings:
                    tokens_embeddings = cast(List[torch.Tensor], tokens_embeddings)  # type: ignore
                    tokens_embeddings = [
                        torch.mean(t, dim=0) for t in tokens_embeddings
                    ]
                    sent_embeddings.append(torch.stack(tokens_embeddings, dim=0))

    if return_embeddings:
        return (sent_tags, sent_embeddings)
    return sent_tags
