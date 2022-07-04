from typing import Any
import unittest, copy
from transformers import BertTokenizerFast  # type: ignore
from hypothesis import assume, given
from conivel.datas import NERSentence, align_tokens_labels_
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


class TestAlignTokensLabels(unittest.TestCase):
    """"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    def test_align_tokens_labels_when_no_tokens_are_decomposed_into_wordpieces(self):
        sent = ["[CLS]", "context", "[SEP]", "hello", "[SEP]", "context", "[SEP]"]
        labels = ["O"] * len(sent)
        words_labels_mask = [0, 0, 0, 1, 0, 0, 0]

        batch = TestAlignTokensLabels.tokenizer(
            sent, is_split_into_words=True, add_special_tokens=False
        )
        batch = align_tokens_labels_(batch, labels, {"O": 0}, words_labels_mask)

        self.assertEqual(batch["labels"], [0] * len(labels))
        self.assertEqual(batch["tokens_labels_mask"], words_labels_mask)

    def test_align_tokens_labels_when_tokens_are_decomposed_into_wordpieces(self):
        # Draculaa will become two wordpieces
        sent = ["[CLS]", "context", "[SEP]", "Draculaa", "[SEP]", "context", "[SEP]"]
        labels = ["O", "O", "O", "B-PER", "O", "O", "O"]
        words_labels_mask = [0, 0, 0, 1, 0, 0, 0]
        labels_dict = {"O": 0, "B-PER": 1}

        batch = TestAlignTokensLabels.tokenizer(
            sent, is_split_into_words=True, add_special_tokens=False
        )
        batch = align_tokens_labels_(batch, labels, labels_dict, words_labels_mask)

        # 1 wordpiece is added (with label B-PER)
        self.assertEqual(batch["labels"], [0, 0, 0, 1, 1, 0, 0, 0])
        # 1 wordpiece is added with a mask value of 1
        self.assertEqual(batch["tokens_labels_mask"], [0, 0, 0, 1, 1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
