from typing import List, Tuple
import re
from pathlib import Path
from conivel.datas.dataset import NERDataset, NERSentence
from conivel.utils import flattened


def _ontonotes_load_block(block: str) -> List[Tuple[str, str]]:
    m = re.match(r"<ENAMEX TYPE=\"([A-Z]*)\">(.*?)</ENAMEX>", block)

    if m is None:
        return [(block, "O")]

    tag, tokens_str = m.groups()
    tokens = tokens_str.split(" ")
    return [(tokens[0], f"B-{tag}")] + [(token, f"I-{tag}") for token in tokens[1:]]


def _ontonotes_split_line(line: str) -> List[str]:
    tokens = []

    chars_buffer = []
    in_tag = 0
    for char in line:
        if char == "<" and in_tag == 0:
            in_tag = 2
        elif char == ">":
            in_tag -= 1
        elif char == " " and in_tag == 0:
            tokens.append("".join(chars_buffer))
            chars_buffer = []
            continue
        chars_buffer.append(char)

    tokens.append("".join(chars_buffer))

    return tokens


def ontonotes_load_document(document_path: Path) -> List[NERSentence]:
    document_test = document_path.read_text()

    ner_sentences = []
    for line in document_test.split("\n")[1:-2]:  # avoid first and last <DOC> lines
        token_and_tag = flattened(
            [_ontonotes_load_block(t) for t in _ontonotes_split_line(line)]
        )
        ner_sentences.append(
            NERSentence(
                [tt[0] for tt in token_and_tag], [tt[1] for tt in token_and_tag]
            )
        )

    return ner_sentences


def ontonotes_load_dir(dir_path: Path) -> List[List[NERSentence]]:
    documents = []
    for path in dir_path.iterdir():
        if str(path).endswith(".name"):
            documents.append(ontonotes_load_document(path))
        elif path.is_dir():
            documents += ontonotes_load_dir(path)
    return documents


class OntonotesDataset(NERDataset):
    def __init__(self, dir_path: str) -> None:
        documents = ontonotes_load_dir(Path(dir_path).expanduser())
        super().__init__(documents)
