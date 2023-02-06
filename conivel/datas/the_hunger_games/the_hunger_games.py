from typing import Optional, List
import os
from conivel.datas import NERSentence
from conivel.datas.dataset import NERDataset
from conivel.utils import flattened


script_dir = os.path.dirname(os.path.abspath(__file__))


class TheHungerGamesDataset(NERDataset):
    """A dataset composed of the first Hunger Games book

    :ivar documents: Either one document (the whole book), or each
        chapter.
    """

    def __init__(
        self, path: Optional[str] = None, cut_into_chapters: bool = True, **kwargs
    ):
        """
        :param cut_into_chapters: if ``True``, each document of
            ``self.documents`` will be a book chapter.  Otherwise,
            ``self.documents`` will consist of a single document, the
            whole book.
        """
        if path is None:
            path = f"{script_dir}/dataset/the_hunger_games.conll"

        documents: List[List[NERSentence]] = [[]]

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

        if not cut_into_chapters:
            documents = [flattened(documents)]

        super().__init__(documents, **kwargs)
