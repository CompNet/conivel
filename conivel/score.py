from typing import List, Set, Tuple, Optional, Union, cast
from seqeval.metrics import precision_score, recall_score, f1_score
from conivel.datas import NERSentence
from conivel.utils import entities_from_bio_tags, flattened


def score_ner(
    ref_sents: Union[List[NERSentence], List[List[str]]],
    pred_bio_tags: List[List[str]],
    ignored_classes: Optional[Set[str]] = None,
    **kwargs
) -> Tuple[float, float, float]:
    """Score NER as in CoNLL-2003 shared task, using ``seqeval``

    Precision is the percentage of named entities in ``ref_bio_tags``
    that are correct. Recall is the percentage of named entities in
    pred_bio_tags that are in ref_bio_tags. F1 is the harmonic mean of
    both.

    :param ref_sents: reference sentences. Can be either a list of
        :class:``bmd.datas.NERSentence`` or a list of sentences tags
        (``List[List[str]]``).

    :param pred_bio_tags: list of sentences prediction tags

    :param ignored_classes: a ``set`` of ignored NER classes
        (example : ``{"LOC", "MISC", "ORG"}``).

    :param kwargs: passed to seqeval functions (``precision_score``,
        ``recall_score`` and ``f1_score``).

    :return: ``(precision, recall, f1-score)``
    """
    assert len(ref_sents) == len(pred_bio_tags)

    ref_tags = ref_sents
    if len(ref_sents) > 0 and all([isinstance(s, NERSentence) for s in ref_sents]):
        ref_sents = cast(List[NERSentence], ref_sents)
        ref_tags = [sent.tags for sent in ref_sents]
    ref_tags = cast(List[List[str]], ref_tags)

    if not ignored_classes is None:
        ref_tags = [
            ["O" if t[2:] in ignored_classes else t for t in sent_tags]
            for sent_tags in ref_tags
        ]
        pred_bio_tags = [
            ["O" if t[2:] in ignored_classes else t for t in sent_tags]
            for sent_tags in pred_bio_tags
        ]

    return (
        cast(float, precision_score(ref_tags, pred_bio_tags, **kwargs)),
        cast(float, recall_score(ref_tags, pred_bio_tags, **kwargs)),
        cast(float, f1_score(ref_tags, pred_bio_tags, **kwargs)),
    )


def score_ner_old(
    ref_sents: List[NERSentence],
    pred_bio_tags: List[List[str]],
    quiet: bool = True,
    ignored_classes: Optional[Set[str]] = None,
    resolve_inconsistencies: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Score NER as in CoNLL-2003 shared task

    Precision is the percentage of named entities in ``ref_bio_tags``
    that are correct. Recall is the percentage of named entities in
    pred_bio_tags that are in ref_bio_tags. F1 is the harmonic mean of
    both.

    .. warning::

        deprecated

    :param ref_sents:
    :param pred_bio_tags:
    :param quiet:
    :param ignored_classes: a `set` of ignored NER classes
        (example : ``{"LOC", "MISC", "ORG"}``).
    :return: ``(precision, recall, F1 score)``
    """
    assert len(pred_bio_tags) == len(ref_sents)

    if len(pred_bio_tags) == 0:
        return (None, None, None)

    tokens = flattened([s.tokens for s in ref_sents])

    pred_entities = entities_from_bio_tags(
        tokens,
        flattened(pred_bio_tags),
        quiet=quiet,
        resolve_inconsistencies=resolve_inconsistencies,
    )
    ref_entities = entities_from_bio_tags(
        tokens,
        flattened([s.tags for s in ref_sents]),
        quiet=quiet,
        resolve_inconsistencies=resolve_inconsistencies,
    )
    if not ignored_classes is None:
        pred_entities = [e for e in pred_entities if not e.tag in ignored_classes]
        ref_entities = [e for e in ref_entities if not e.tag in ignored_classes]

    # TODO: optim
    correct_predictions = 0
    for pred_entity in pred_entities:
        if pred_entity in ref_entities:
            correct_predictions += 1
    precision = None
    if len(pred_entities) > 0:
        precision = correct_predictions / len(pred_entities)

    # TODO: optim
    recalled_entities = 0
    for ref_entity in ref_entities:
        if ref_entity in pred_entities:
            recalled_entities += 1
    recall = None
    if len(ref_entities) > 0:
        recall = recalled_entities / len(ref_entities)

    if precision is None or recall is None or precision + recall == 0:
        return (precision, recall, None)
    f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)
