from string import ascii_letters
from hypothesis import strategies as st
from hypothesis.strategies import composite
from conivel.datas.datas import NERSentence


@composite
def ner_sentence(draw, min_len: int = 0, max_len: int = 100) -> NERSentence:
    """A strategies that generate ner sentences

    .. note::

        generated sentences do not have left or right contexts.


    :param min_len: min size of the generated ``NERSentence``
    :param max_len: max size of the generated ``NERSentence``

    :return: a generated ``NERSentence``
    """
    sent_len = draw(st.integers(min_value=min_len, max_value=max_len))
    tokens = draw(
        st.lists(st.sampled_from(ascii_letters), min_size=sent_len, max_size=sent_len)
    )
    tags = draw(
        st.lists(
            st.sampled_from(["O", "B-PER", "I-PER"]),
            min_size=sent_len,
            max_size=sent_len,
        )
    )
    return NERSentence(tokens, tags)
