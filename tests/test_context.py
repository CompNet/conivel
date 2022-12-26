import os
from typing import List
import unittest
from hypothesis import given, settings
from hypothesis.control import assume
import hypothesis.strategies as st
from hypothesis.strategies import composite
from conivel.datas.context import (
    ContextRetrievalExample,
    ContextRetrievalDataset,
    NeuralContextRetriever,
)


@composite
def context_retrieval_example(draw):
    sent = draw(st.lists(st.text()))
    context = draw(st.lists(st.text(), min_size=len(sent), max_size=len(sent)))
    context_side = draw(st.sampled_from(["left", "right"]))
    usefulness = draw(st.floats(min_value=-1, max_value=1))
    sent_was_correctly_predicted = draw(st.booleans())
    return ContextRetrievalExample(
        sent, context, context_side, usefulness, sent_was_correctly_predicted
    )


class TestNeuralContextRetriever(unittest.TestCase):
    """"""

    @unittest.skipIf(os.getenv("CONIVEL_TEST_ALL") != "1", "skipped for performance")
    @settings(deadline=None)
    @given(examples=st.lists(context_retrieval_example(), min_size=1))
    def test_balance_context_dataset(self, examples: List[ContextRetrievalExample]):
        assume(any([ex.sent_was_correctly_predicted for ex in examples]))
        assume(any([not ex.sent_was_correctly_predicted for ex in examples]))
        dataset = ContextRetrievalDataset(examples)
        balanced_dataset = NeuralContextRetriever.balance_context_dataset(
            dataset, bins_nb=10
        )
        self.assertLessEqual(len(balanced_dataset), len(examples))


if __name__ == "__main__":
    unittest.main()
