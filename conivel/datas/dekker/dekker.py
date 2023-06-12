from typing import List, Optional, Set
import os, glob, re
from collections import Counter
import nltk
from tqdm import tqdm
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

        paths = sorted(glob.glob(f"{directory}/*.conll"))

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
            documents_attrs.append({"name": os.path.basename(book_path)})

        super().__init__(documents, documents_attrs=documents_attrs, **kwargs)


def load_extended_documents(
    directory: str, dataset: DekkerDataset
) -> List[List[NERSentence]]:
    """Load dekker's dataset full documents.  Use for retrieval
    purposes only, as tags wont be included.

    :param directory: Directory containing the extended documents.
        They must have the same name as documents in the dekker
        dataset, and end in .txt
    :param dataset:

    :return:
    """

    dir_files = sorted(glob.glob(f"{directory}/*.txt"))

    extended_documents = []

    for doc, doc_attrs in tqdm(
        zip(dataset.documents, dataset.documents_attrs), total=len(dataset.documents)
    ):
        # NOTE: this is because names in DekkerDataset have the form
        # 'title.conll`, but we only care about the title! Kind of a
        # hack.
        doc_name = os.path.splitext(doc_attrs["name"])[0]

        # find corresponding file
        dir_file = None
        for df in dir_files:
            df_name = os.path.splitext(os.path.basename(df))[0]
            if df_name == doc_name:
                dir_file = df
                break
        assert not dir_file is None

        with open(dir_file) as f:
            extended_sents = [
                nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(f.read())
            ]

        # Loosely (loosely !) find the end of doc in its extended
        # sents. Kinda hacky but it seems to work for the most part.
        ex_sents_counter = Counter([tuple(s) for s in extended_sents])
        for i in range(1, len(doc)):
            last_sent = doc[-i].tokens
            # sentence must be unique
            if not ex_sents_counter.get(tuple(last_sent), 0) == 1:
                continue
            try:
                last_sent_index = extended_sents.index(last_sent)
                break
            except ValueError:
                continue

        extension_sents = [
            NERSentence(tokens, ["O"] * len(tokens))
            for tokens in extended_sents[last_sent_index:]
        ]
        extended_documents.append(doc + extension_sents)

    return extended_documents
