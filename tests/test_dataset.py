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
        dataset = NERDataset([[NERSentence([], [])] * sents_nb], tokenizer=0)
        kfolds = dataset.kfolds(5)
        self.assertEqual(len(kfolds), 5)
        self.assertTrue(all([len(sets) == 2 for sets in kfolds]))
