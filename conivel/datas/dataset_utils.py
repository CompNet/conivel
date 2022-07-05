from typing import Generator
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from conivel.datas import DataCollatorForTokenClassificationWithBatchEncoding
from conivel.datas.dataset import NERDataset


def dataset_batchs(
    dataset: NERDataset, batch_size: int, shuffle: bool = False, quiet: bool = False
) -> Generator[BatchEncoding, None, None]:
    data_collator = DataCollatorForTokenClassificationWithBatchEncoding(
        dataset.tokenizer  # type: ignore
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle
    )

    for batch in tqdm(dataloader, disable=quiet):
        yield batch
