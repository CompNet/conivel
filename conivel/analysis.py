from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from dataclasses import dataclass

from conivel.datas.datas import NERSentence
from conivel.utils import entities_from_bio_tags


@dataclass
class PredictionError:
    sent: NERSentence
    start_idx: int
    end_idx: int
    pred: List[str]
    error_type: Literal["precision", "recall"]

    @property
    def ref(self) -> List[str]:
        """Original tags"""
        return self.sent.tags[self.start_idx : self.end_idx + 1]

    @property
    def pred_class(self) -> Optional[str]:
        """Predicted class.

        :return: the class string, or ``None`` if the class is unclear
                 (for example, in the case of several different
                 classes for a group of tokens)
        """
        if all([p == "O" for p in self.pred]):
            return "O"
        elif len({p[2:] for p in self.pred}) == 1:
            return self.pred[0][:2]
        return None

    @property
    def ref_class(self) -> str:
        if self.sent.tags[self.start_idx] == "O":
            return "O"
        return self.sent.tags[self.start_idx][2:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sent": {"tokens": self.sent.tokens, "tags": self.sent.tags},
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "pred": self.pred,
            "error_type": self.error_type,
        }

    @staticmethod
    def from_dict(self, d: Dict[str, Any]) -> PredictionError:
        sent = NERSentence(d["sent"]["tokens"], d["sent"]["tags"])
        del d["sent"]
        return PredictionError(sent, **d)


def get_errors(ref_sent: NERSentence, pred_tags: List[str]) -> List[PredictionError]:
    """Retrieve the errors for a prediction.

    :param ref_sent: reference :class:`NERSentence`.
    :param pred_tags: prediction, should be of the same length as ``ref_sent``.

    :return: a list of :class:`PredictionError`.  Each error can span
             multiple contiguous tokens.
    """
    assert len(ref_sent) == len(pred_tags)

    ref_entities = entities_from_bio_tags(ref_sent.tokens, ref_sent.tags)
    pred_entities = entities_from_bio_tags(ref_sent.tokens, pred_tags)

    errors = []

    for ref_entity in ref_entities:
        if not ref_entity in pred_entities:
            errors.append(
                PredictionError(
                    ref_sent,
                    ref_entity.start_idx,
                    ref_entity.end_idx,
                    pred_tags[ref_entity.start_idx : ref_entity.end_idx + 1],
                    error_type="recall",
                )
            )

    for pred_entity in pred_entities:
        if not pred_entity in ref_entities:
            errors.append(
                PredictionError(
                    ref_sent,
                    pred_entity.start_idx,
                    pred_entity.end_idx,
                    pred_tags[pred_entity.start_idx : pred_entity.end_idx + 1],
                    error_type="precision",
                )
            )

    return errors
