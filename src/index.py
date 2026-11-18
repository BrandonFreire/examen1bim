"""Índice invertido en memoria."""

from __future__ import annotations

from collections import Counter

from src.data import Doc
from src.preprocessing import preprocess


class InvertedIndex:
    """postings[término][doc_id] = tf, df[término] y longitudes de documento."""

    def __init__(self, *, stem: bool = False) -> None:
        self.stem = stem
        self.postings: dict[str, dict[str, int]] = {}
        self.df: dict[str, int] = {}
        self.doc_lengths: dict[str, int] = {}
        self.N = 0

    @classmethod
    def build(cls, corpus: dict[str, Doc], *, stem: bool = False) -> InvertedIndex:
        index = cls(stem=stem)
        index.N = len(corpus)

        for doc_id, doc in corpus.items():
            tokens = preprocess(f"{doc.title} {doc.text}", stem=stem)
            index.doc_lengths[doc_id] = len(tokens)

            for term, tf in Counter(tokens).items():
                index.postings.setdefault(term, {})[doc_id] = tf
                index.df[term] = index.df.get(term, 0) + 1

        return index
