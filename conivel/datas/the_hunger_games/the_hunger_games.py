from typing import Optional
import os
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset


script_dir = os.path.dirname(os.path.abspath(__file__))


class TheHungerGamesDataset(NERDataset):
    """A dataset composed of the first Hunger Games book

    :ivar documents: Each document represent a paragraph, composed of
        a list of sents
    """

    def __init__(self, path: Optional[str] = None, **kwargs):
        """"""
        if path is None:
            path = f"{script_dir}/dataset/the_hunger_games.conll"

        documents = [[]]

        with open(path) as f:

            sent = NERSentence([], [])
            in_quote = False
            prev_line_was_space = False

            for line in f:

                # cut into chapters
                if line.isspace():
                    if len(sent) > 0:
                        documents[-1].append(sent)
                        sent = NERSentence([], [])
                    if prev_line_was_space:
                        documents.append([])
                        continue
                    prev_line_was_space = True
                    continue
                prev_line_was_space = False

                token, tag = line.strip().split("\t")

                sent.tokens.append(token)
                sent.tags.append(tag)

                if token == '"':
                    if in_quote:
                        documents[-1].append(sent)
                        sent = NERSentence([], [])
                    in_quote = not in_quote

                elif token in {".", "?", "!"} and not in_quote:
                    documents[-1].append(sent)
                    sent = NERSentence([], [])

        # chapter 0 is the title page -> we remove it
        documents = documents[1:]

        super().__init__(documents, **kwargs)
