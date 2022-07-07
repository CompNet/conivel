from string import ascii_letters
from hypothesis import strategies as st
from hypothesis.strategies import composite
from conivel.datas.datas import NERSentence


@composite
def ner_sentence(
    draw,
    min_len: int = 0,
    max_len: int = 100,
    left_ctx_min_nb: int = 0,
    left_ctx_max_nb: int = 0,
    right_ctx_min_nb: int = 0,
    right_ctx_max_nb: int = 0,
) -> NERSentence:
    """A strategies that generate ner sentences

    :param min_len: min size of generated ``NERSentence``
    :param max_len: max size of generated ``NERSentence``

    :param left_ctx_min_nb: min number of left context sentences to
        generate
    :param left_ctx_max_nb: max number of left context sentences to
        generate

    :param right_ctx_min_nb: min number of right context sentences to
        generate
    :param right_ctx_max_nb: max number of right context sentences to
        generate

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

    left_ctx = [
        draw(ner_sentence(min_len, max_len))
        for _ in range(left_ctx_min_nb, left_ctx_max_nb)
    ]

    right_ctx = [
        draw(ner_sentence(min_len, max_len))
        for _ in range(left_ctx_min_nb, left_ctx_max_nb)
    ]

    return NERSentence(tokens, tags, left_ctx, right_ctx)
