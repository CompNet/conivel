import unittest, copy
from hypothesis import assume, given
from conivel.datas import NERSentence
from strategies import ner_sentence


class TestNERSentenceHashing(unittest.TestCase):
    """"""

    @given(sent=ner_sentence())
    def test_equal_sents_have_equal_hashes(self, sent: NERSentence):
        self.assertEqual(hash(sent), hash(copy.deepcopy(sent)))

    @given(sent1=ner_sentence(), sent2=ner_sentence())
    def test_unequal_sents_have_different_hashes(
        self, sent1: NERSentence, sent2: NERSentence
    ):
        assume(sent1 != sent2)
        self.assertNotEqual(hash(sent1), hash(sent2))
