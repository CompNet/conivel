from __future__ import annotations
from typing import Optional, Set
import os
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset

script_dir = os.path.dirname(os.path.abspath(__file__))


class CoNLLDataset(NERDataset):
    """A class representing a CoNLL-2003 dataset"""

    def __init__(
        self,
        path: str,
        keep_only_classes: Optional[Set[str]] = None,
        separator: str = " ",
        **kwargs,
    ) -> None:
        """
        :param path: dataset file path
        :param keep_only_classes: if not ``None``, keep only NER
            classes from this set
        :param separator: separator between tokens and tags
        """
        # Dataset loading
        with open(path) as f:
            raw_datas = f.read().strip()

        self.documents = []

        for raw_document in raw_datas.split("-DOCSTART- O\n\n"):

            if raw_document == "":
                continue

            self.documents.append([])

            for sent in raw_document.split("\n\n"):

                if sent == "":
                    continue

                self.documents[-1].append(NERSentence([], []))

                for line in sent.split("\n"):

                    # tokens parsing
                    self.documents[-1][-1].tokens.append(line.split(separator)[0])

                    # tags parsing
                    tag = line.split(separator)[1]
                    if keep_only_classes:
                        self.documents[-1][-1].tags.append(
                            tag if tag[2:] in keep_only_classes else "O"
                        )
                    else:
                        self.documents[-1][-1].tags.append(tag)

        tags = set(
            [
                tag
                for document in self.documents
                for sent in document
                for tag in sent.tags
            ]
        )

        # Init
        super().__init__(self.documents, tags, **kwargs)

    @staticmethod
    def train_dataset(**kwargs) -> CoNLLDataset:
        return CoNLLDataset(f"{script_dir}/train2.txt", **kwargs)

    @staticmethod
    def test_dataset(
        **kwargs,
    ) -> CoNLLDataset:
        return CoNLLDataset(f"{script_dir}/test2.txt", **kwargs)

    @staticmethod
    def valid_dataset(**kwargs) -> CoNLLDataset:
        return CoNLLDataset(f"{script_dir}/valid2.txt", **kwargs)
