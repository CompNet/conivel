from typing import cast
import unittest
import torch
from transformers import BertTokenizerFast, BertForTokenClassification  # type: ignore
from conivel.predict import _get_batch_tags
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset
from conivel.datas.dataset_utils import dataset_batchs


class TestBatchParsing(unittest.TestCase):
    """"""

    @classmethod
    def setUpClass(cls) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cls.model = BertForTokenClassification.from_pretrained("bert-base-cased")
        cls.model = cast(BertForTokenClassification, cls.model)
        cls.model = cls.model.to(device)
        cls.model = cls.model.eval()

        cls.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    def test_batch_tags_extraction(self):
        batch_size = 2

        sents = [NERSentence(["sentence"], ["O"])] * 3

        dataset = NERDataset([sents])

        with torch.no_grad():

            for batch_i, batch in enumerate(dataset_batchs(dataset, batch_size)):

                out = TestBatchParsing.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )  # type: ignore

                tags = _get_batch_tags(
                    batch, out.logits, TestBatchParsing.model.config.id2label  # type: ignore
                )

                batch_sents = sents[batch_i * batch_size : batch_size * (batch_i + 1)]
                for sent_tags, sent in zip(tags, batch_sents):
                    self.assertEqual(len(sent_tags), len(sent))


if __name__ == "__main__":
    unittest.main()
