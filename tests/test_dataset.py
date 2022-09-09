import unittest
from hypothesis import given
from hypothesis.strategies import integers
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset


class TestNERDataset(unittest.TestCase):
    """"""

    @given(sents_nb=integers(min_value=0, max_value=30))
    def test_kfolds(self, sents_nb: int):
        # HACK: replace tokenizer by int to avoid loading it - it wont be used anyway
        dataset = NERDataset([[NERSentence([], [])] * sents_nb], tokenizer=0)  # type: ignore
        kfolds = dataset.kfolds(5)
        self.assertEqual(len(kfolds), 5)
        self.assertTrue(all([len(sets) == 2 for sets in kfolds]))

    @given(
        sents_nb1=integers(min_value=0, max_value=30),
        sents_nb2=integers(min_value=0, max_value=30),
    )
    def test_concatenated(self, sents_nb1: int, sents_nb2: int):
        # HACK: replace tokenizer by int to avoid loading it - it wont be used anyway
        dataset1 = NERDataset([[NERSentence([], [])] * sents_nb1], tokenizer=0)  # type: ignore
        dataset2 = NERDataset([[NERSentence([], [])] * sents_nb2], tokenizer=0)  # type: ignore
        concatenated = NERDataset.concatenated([dataset1, dataset2])
        self.assertEqual(len(concatenated), sents_nb1 + sents_nb2)
