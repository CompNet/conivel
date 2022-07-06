from typing import cast
import unittest
from hypothesis import given, settings
from hypothesis.strategies import integers
import torch
from transformers import BertTokenizerFast, BertForTokenClassification  # type: ignore
from conivel.predict import (
    _get_batch_tags,
    _get_batch_embeddings,
    _get_batch_scores,
    _get_batch_attentions,
)
from conivel.utils import get_tokenizer
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset
from conivel.datas.dataset_utils import dataset_batchs
from strategies import ner_sentence


class TestBatchParsing(unittest.TestCase):
    """"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = get_tokenizer()

    @settings(deadline=None)  # deactivate deadline because of tokenizer instantiation
    @given(
        sent=ner_sentence(min_len=1, max_len=25),
        sents_nb=integers(min_value=1, max_value=100),
        batch_size=integers(min_value=1, max_value=16),
    )
    def test_batch_tags_extraction(
        self, sent: NERSentence, sents_nb: int, batch_size: int
    ):
        sents = [sent] * sents_nb
        dataset = NERDataset([sents], tokenizer=TestBatchParsing.tokenizer)

        for batch_i, batch in enumerate(
            dataset_batchs(dataset, batch_size, quiet=True)
        ):
            l_batch_size, seq_size = batch["input_ids"].shape  # type: ignore

            # perfect predictions
            logits = torch.zeros(l_batch_size, seq_size, dataset.tags_nb)
            for i in range(l_batch_size):
                for j in range(seq_size):
                    logits[i][j][batch["labels"][i][j]] = 1  # type: ignore

            pred_tags = _get_batch_tags(batch, logits, dataset.id_to_tag)

            batch_sents = sents[batch_i * batch_size : batch_size * (batch_i + 1)]
            for tags, sent in zip(pred_tags, batch_sents):
                self.assertEqual(tags, sent.tags)

    @given(
        sent=ner_sentence(min_len=1, max_len=25),
        sents_nb=integers(min_value=1, max_value=100),
        batch_size=integers(min_value=1, max_value=16),
    )
    def test_batch_embeddings_extraction(
        self, sent: NERSentence, sents_nb: int, batch_size: int
    ):
        sents = [sent] * sents_nb
        dataset = NERDataset([sents], tokenizer=TestBatchParsing.tokenizer)
        hidden_size = 10

        for batch_i, batch in enumerate(
            dataset_batchs(dataset, batch_size, quiet=True)
        ):
            l_batch_size, seq_size = batch["input_ids"].shape  # type: ignore

            embeddings = torch.zeros(l_batch_size, seq_size, hidden_size)
            pred_embeddings = _get_batch_embeddings(batch, embeddings)

            batch_sents = sents[batch_i * batch_size : batch_size * (batch_i + 1)]
            for emb, sent in zip(pred_embeddings, batch_sents):
                self.assertEqual((len(sent), hidden_size), emb.shape)

    @given(
        sent=ner_sentence(min_len=1, max_len=25),
        sents_nb=integers(min_value=1, max_value=100),
        batch_size=integers(min_value=1, max_value=16),
    )
    def test_batch_scores_extraction(
        self, sent: NERSentence, sents_nb: int, batch_size: int
    ):
        sents = [sent] * sents_nb
        dataset = NERDataset([sents], tokenizer=TestBatchParsing.tokenizer)

        for batch_i, batch in enumerate(
            dataset_batchs(dataset, batch_size, quiet=True)
        ):
            l_batch_size, seq_size = batch["input_ids"].shape  # type: ignore

            logits = torch.zeros(l_batch_size, seq_size, dataset.tags_nb)

            pred_scores = _get_batch_scores(batch, logits)

            batch_sents = sents[batch_i * batch_size : batch_size * (batch_i + 1)]
            for scores, sent in zip(pred_scores, batch_sents):
                self.assertEqual((len(sent),), scores.shape)

    def test_batch_attentions_extraction(self):
        pass


if __name__ == "__main__":
    unittest.main()
