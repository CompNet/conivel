from typing import List
import unittest
from hypothesis import given, settings
from hypothesis.strategies import integers, lists
import torch
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
        sents=lists(
            ner_sentence(min_len=1, max_len=16, left_ctx_max_nb=2, right_ctx_max_nb=2),
            min_size=2,
            max_size=16,
        ),
        batch_size=integers(min_value=1, max_value=8),
    )
    def test_batch_tags_extraction(self, sents: List[NERSentence], batch_size: int):
        dataset = NERDataset([sents], tokenizer=TestBatchParsing.tokenizer)

        for batch_i, batch in enumerate(
            dataset_batchs(dataset, batch_size, quiet=True)
        ):
            l_batch_size, seq_size = batch["input_ids"].shape  # type: ignore

            # perfect predictions
            logits = torch.zeros(l_batch_size, seq_size, dataset.tags_nb)
            for i in range(l_batch_size):
                for j in range(seq_size):
                    # ignore padding
                    if batch["labels"][i][j] == -100:  # type: ignore
                        continue
                    logits[i][j][batch["labels"][i][j]] = 1  # type: ignore

            pred_tags = _get_batch_tags(batch, logits, dataset.id_to_tag)

            batch_sents = sents[batch_i * batch_size : batch_size * (batch_i + 1)]
            for tags, sent in zip(pred_tags, batch_sents):
                self.assertEqual(tags, sent.tags)

    @settings(deadline=None)
    @given(
        sents=lists(
            ner_sentence(min_len=1, max_len=16, left_ctx_max_nb=2, right_ctx_max_nb=2),
            min_size=1,
            max_size=16,
        ),
        batch_size=integers(min_value=1, max_value=8),
    )
    def test_batch_embeddings_extraction(
        self, sents: List[NERSentence], batch_size: int
    ):
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

    @settings(deadline=None)
    @given(
        sents=lists(
            ner_sentence(min_len=1, max_len=16, left_ctx_max_nb=2, right_ctx_max_nb=2),
            min_size=1,
            max_size=16,
        ),
        batch_size=integers(min_value=1, max_value=8),
    )
    def test_batch_scores_extraction(self, sents: List[NERSentence], batch_size: int):
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

    @settings(deadline=None)
    @given(
        sents=lists(
            ner_sentence(min_len=1, max_len=16, left_ctx_max_nb=2, right_ctx_max_nb=2),
            min_size=1,
            max_size=16,
        ),
        batch_size=integers(min_value=1, max_value=8),
    )
    def test_batch_attentions_extraction(
        self, sents: List[NERSentence], batch_size: int
    ):
        dataset = NERDataset([sents], tokenizer=TestBatchParsing.tokenizer)
        layers_nb = 12
        heads_nb = 8

        for batch_i, batch in enumerate(
            dataset_batchs(dataset, batch_size, quiet=True)
        ):
            l_batch_size, seq_size = batch["input_ids"].shape  # type: ignore

            attentions = torch.zeros(
                layers_nb, heads_nb, l_batch_size, seq_size, seq_size
            )

            batch_attentions = _get_batch_attentions(batch, attentions)

            batch_sents = sents[batch_i * batch_size : batch_size * (batch_i + 1)]
            for att, sent in zip(batch_attentions, batch_sents):
                special_tokens_len = 2
                if len(sent.left_context) > 0:
                    special_tokens_len += 1
                if len(sent.right_context) > 0:
                    special_tokens_len += 1
                self.assertEqual(
                    (
                        layers_nb,
                        heads_nb,
                        sent.len_with_ctx() + special_tokens_len,
                        sent.len_with_ctx() + special_tokens_len,
                    ),
                    att.shape,
                )


if __name__ == "__main__":
    unittest.main()
