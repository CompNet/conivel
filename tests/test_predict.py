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
        cls.model = cls.model.to(device)
        cls.model = cls.model.eval()

        cls.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    def test_batch_tags_extraction(self):

        dataset = NERDataset(
            [
                [
                    NERSentence(["left", "context"], ["O", "O"]),
                    NERSentence(["sentence"], ["O"]),
                    NERSentence(["right", "context"], ["O", "O"]),
                ]
            ]
        )

        with torch.no_grad():

            for batch in dataset_batchs(dataset, 2):

                out = TestBatchParsing.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )

                tags = _get_batch_tags(
                    batch, out.logits, TestBatchParsing.model.config.id2label
                )


if __name__ == "__main__":
    unittest.main()
