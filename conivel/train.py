from typing import Optional, List, Set, cast
import copy, json
from sacred.run import Run
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from transformers import BertForTokenClassification
from conivel.datas import (
    NERDataset,
    batch_to_device,
    DataCollatorForTokenClassificationWithBatchEncoding,
)
from conivel.predict import predict
from conivel.score import score_ner
from conivel.utils import flattened, get_tokenizer


def train_ner_model(
    model: BertForTokenClassification,
    train_dataset: NERDataset,
    valid_dataset: NERDataset,
    _run: Optional[Run] = None,
    epochs_nb: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    use_class_weights: bool = False,
    custom_weights: Optional[List[float]] = None,
    quiet: bool = False,
    ignored_valid_classes: Optional[Set[str]] = None,
) -> BertForTokenClassification:
    """
    :param train_dataset:
    :param custom_weights: A list of class weights. Should be ordered by
        using ``dataset.tag_to_id``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    tokenizer = get_tokenizer()

    weights = None
    if use_class_weights:
        weights = torch.tensor(train_dataset.tag_weights()).to(device)
    elif custom_weights:
        weights = torch.tensor(custom_weights).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    data_collator = DataCollatorForTokenClassificationWithBatchEncoding(tokenizer)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    best_f1 = 0
    best_model = model

    for epoch in range(epochs_nb):

        epoch_losses = []
        model = model.train()

        data_tqdm = tqdm(dataloader) if not quiet else dataloader
        for X in data_tqdm:

            optimizer.zero_grad()

            X = batch_to_device(X, device)

            # (batch_size, seq_size, tags_nb)
            out = model(
                X["input_ids"],
                token_type_ids=X["token_type_ids"],
                attention_mask=X["attention_mask"],
            )

            loss = loss_fn(out.logits.permute(0, 2, 1), X["labels"])

            loss.backward()
            optimizer.step()

            if not quiet:
                data_tqdm.set_description(f"loss : {loss.item():.3f}")
            if not _run is None:
                _run.log_scalar("training.loss", loss.item())
            epoch_losses.append(loss.item())

        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        tqdm.write(f"epoch mean loss : {mean_epoch_loss:.3f}")
        if not _run is None:
            _run.log_scalar("training.mean_epoch_loss", mean_epoch_loss)

        # compute, record and print validation metrics
        predicted_tags = predict(
            model, valid_dataset, batch_size=batch_size, quiet=quiet
        )
        predicted_tags = cast(List[List[str]], predicted_tags)
        precision, recall, f1 = score_ner(
            valid_dataset.sents(), predicted_tags, ignored_classes=ignored_valid_classes
        )

        if not _run is None:
            _run.log_scalar("training.validation_precision", precision)
            _run.log_scalar("training.validation_recall", recall)
            _run.log_scalar("training.validation_f1", f1)

        tqdm.write(
            json.dumps(
                {
                    "validation precision": precision,
                    "validation recall": recall,
                    "validation f1": f1,
                },
                indent=4,
            )
        )

        if not f1 is None and f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model)

    return best_model
