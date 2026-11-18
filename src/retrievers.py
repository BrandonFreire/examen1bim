"""Recuperadores TF-IDF (coseno) y BM25 sobre el índice invertido propio."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log, log1p, sqrt
from typing import Protocol

from src.index import InvertedIndex
from src.preprocessing import preprocess


@dataclass(frozen=True)
class Result:
    doc_id: str
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 10) -> list[Result]: ...


def _top_k(scores: dict[str, float], k: int) -> list[Result]:
    results = [Result(doc_id, score) for doc_id, score in scores.items()]
    results.sort(key=lambda r: (-r.score, r.doc_id))
    return results[:k]


class TfidfRetriever:
    """tf = 1 + ln(f), idf = ln(N/df) y similitud coseno (norma L2)."""

    def __init__(self, index: InvertedIndex, stem: bool = False) -> None:
        if index.stem != stem:
            raise ValueError("El índice tiene una configuración de stemming distinta")

        self.index = index
        self.stem = stem
        self.idf = {term: log(index.N / df) for term, df in index.df.items()}

        squared_norms: Counter[str] = Counter()
        for term, postings in index.postings.items():
            idf = self.idf[term]
            if idf == 0:
                continue
            for doc_id, tf in postings.items():
                squared_norms[doc_id] += ((1 + log(tf)) * idf) ** 2
        self.doc_norms = {doc_id: sqrt(sq) for doc_id, sq in squared_norms.items()}

    def search(self, query: str, k: int = 10) -> list[Result]:
        if k <= 0:
            raise ValueError("k debe ser un entero positivo")

        frequencies = Counter(preprocess(query, stem=self.stem))
        query_weights = {
            term: (1 + log(tf)) * self.idf[term]
            for term, tf in frequencies.items()
            if self.idf.get(term, 0) > 0
        }
        query_norm = sqrt(sum(w * w for w in query_weights.values()))
        if query_norm == 0:
            return []

        dots: Counter[str] = Counter()
        for term, q_weight in query_weights.items():
            idf = self.idf[term]
            for doc_id, tf in self.index.postings[term].items():
                dots[doc_id] += q_weight * (1 + log(tf)) * idf

        scores = {
            doc_id: dot / (query_norm * self.doc_norms[doc_id])
            for doc_id, dot in dots.items()
        }
        return _top_k(scores, k)


class BM25Retriever:
    """BM25 con k1=1.5, b=0.75 e idf = ln(1 + (N-df+0.5)/(df+0.5))."""

    def __init__(self, index: InvertedIndex, stem: bool = False) -> None:
        if index.stem != stem:
            raise ValueError("El índice tiene una configuración de stemming distinta")

        self.index = index
        self.stem = stem
        self.k1 = 1.5
        self.b = 0.75
        self.avgdl = sum(index.doc_lengths.values()) / index.N if index.N else 0.0
        self.idf = {
            term: log1p((index.N - df + 0.5) / (df + 0.5))
            for term, df in index.df.items()
        }
        # k1 * (1 - b + b * dl / avgdl): parte de la fórmula que depende solo del documento
        self.length_norm = (
            {
                doc_id: self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                for doc_id, dl in index.doc_lengths.items()
            }
            if self.avgdl > 0
            else {}
        )

    def search(self, query: str, k: int = 10) -> list[Result]:
        if k <= 0:
            raise ValueError("k debe ser un entero positivo")
        if self.avgdl == 0:
            return []

        scores: Counter[str] = Counter()
        for term in sorted(set(preprocess(query, stem=self.stem))):
            postings = self.index.postings.get(term)
            if postings is None:
                continue
            idf = self.idf[term]
            for doc_id, tf in postings.items():
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.length_norm[doc_id])

        return _top_k(scores, k)
