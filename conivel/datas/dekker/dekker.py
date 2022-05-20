from typing import List, Optional
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


class DekkerDataset(NERDataset):
    """"""

    def __init__(
        self,
        directory: Optional[str] = None,
        book_group: Optional[str] = None,
        **kwargs,
    ):
        """"""
        if directory is None:
            directory = f"{script_dir}/dataset"

        new_paths = glob.glob(f"{directory}/new/*.conll.fixed")
        old_paths = glob.glob(f"{directory}/old/*.conll.fixed")

        def book_name(path: str) -> str:
            return re.search(r"[^.]*", (os.path.basename(path))).group(0)  # type: ignore

        documents = []

        for book_path in new_paths + old_paths:

            cur_doc = []

            if not book_group is None:
                name = book_name(book_path)
                if not name in book_groups[book_group]:
                    continue

            with open(book_path) as f:

                sent = NERSentence([], [])
                in_quote = False

                for line in f:

                    splitted = line.strip().split(" ")

                    sent.tokens.append(splitted[0])
                    sent.tags.append(splitted[1])

                    if splitted[0] == "``":
                        in_quote = True
                    elif splitted[0] == "''":
                        in_quote = False
                        cur_doc.append(sent)
                        sent = NERSentence([], [])
                    elif splitted[0] in {".", "?", "!"} and not in_quote:
                        cur_doc.append(sent)
                        sent = NERSentence([], [])

            documents.append(cur_doc)

        super().__init__(documents, **kwargs)
