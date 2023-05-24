from typing import List, Optional, Set
import os, glob, re
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset


script_dir = os.path.dirname(os.path.abspath(__file__))

book_groups = {
    "fantasy": {
        "TheFellowshipoftheRing",
        "TheWheelOfTime",
        "TheWayOfShadows",
        "TheBladeItself",
        "Elantris",
        "ThePaintedMan",
        "GardensOfTheMoon",
        "Magician",
        "BlackPrism",
        "TheBlackCompany",
        "Mistborn",
        "AGameOfThrones",
        "AssassinsApprentice",
        "TheNameOfTheWind",
        "TheColourOfMagic",
        "TheWayOfKings",
        "TheLiesOfLockeLamora",
    }
}


def load_book(
    path: str, keep_only_classes: Optional[Set[str]] = None
) -> List[NERSentence]:
    """
    :param path: book path
    :param keep_only_classes: if not ``None``, only keep tags from the
        given NER classes
    """

    # load tokens and tags from CoNLL formatted file
    tokens = []
    tags = []

    with open(path) as f:
        for i, line in enumerate(f):
            try:
                token, tag = line.strip().split(" ")
            except ValueError:
                print(f"error processing line {i+1} of book {path}")
                print(f"line content was : '{line}'")
                print("trying to proceed...")
                continue

            if not keep_only_classes is None and not tag == "O":
                tag = tag if tag[2:] in keep_only_classes else "O"

            tokens.append(token)
            tags.append(tag)

    # parse into sentences
    sents = []
    sent = NERSentence()

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        fixed_token = '"' if token in {"``", "''"} else token
        fixed_token = "'" if token == "`" else fixed_token
        next_token = tokens[i + 1] if i < len(tokens) - 1 else None

        sent.tokens.append(fixed_token)
        sent.tags.append(tag)

        # quote ends next token : skip this token
        # this avoids problem with cases where we have punctuation
        # at the end of a quote (otherwise, the end of the quote
        # would be part of the next sentence)
        if next_token == "''":
            continue

        # sentence end
        if token in ["''", ".", "?", "!"]:
            sents.append(sent)
            sent = NERSentence()

    return sents


class DekkerDataset(NERDataset):
    """"""

    def __init__(
        self,
        directory: Optional[str] = None,
        book_group: Optional[str] = None,
        keep_only_classes: Optional[Set[str]] = None,
        **kwargs,
    ):
        """"""
        if directory is None:
            directory = f"{script_dir}/dataset"

        paths = glob.glob(f"{directory}/*.conll")

        def book_name(path: str) -> str:
            return re.search(r"[^.]*", (os.path.basename(path))).group(0)  # type: ignore

        documents = []
        documents_attrs = []

        for book_path in paths:
            # skip book if it's not in the given book group
            if not book_group is None:
                name = book_name(book_path)
                if not name in book_groups[book_group]:
                    continue

            documents.append(load_book(book_path, keep_only_classes=keep_only_classes))
            self.documents_attrs.append({"name": os.path.basename(book_path)})

        super().__init__(documents, documents_attrs=documents_attrs, **kwargs)
