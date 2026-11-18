"""Preprocesamiento léxico en inglés para documentos y consultas de NFCorpus."""

from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

_NLTK_RESOURCES = (
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
)
_SEPARATORS = re.compile(r"[\W_]+")
_stemmer = PorterStemmer()


def setup_nltk() -> None:
    """Descarga los recursos de NLTK solo si todavía no están instalados."""
    for resource_path, package_name in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset[str]:
    setup_nltk()
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=100_000)
def _stem(token: str) -> str:
    return _stemmer.stem(token)


def alnum_pieces(token: str) -> list[str]:
    """Trozos alfanuméricos de un token: 'beta-carotene' -> 'beta', 'carotene'."""
    if token.isalnum():
        return [token]
    if token == "n't":
        return []
    return [piece for piece in _SEPARATORS.split(token) if piece]


def preprocess(text: str, *, stem: bool = False) -> list[str]:
    """Tokeniza, pasa a minúsculas, quita puntuación y stopwords; stemming opcional.

    Los números se conservan (ej. '5', 'b12') porque en textos médicos suelen
    ser informativos (vitamina b12, omega 3).
    """
    if not text or not text.strip():
        return []

    stop_words = _get_stopwords()
    tokens = [
        piece
        for token in word_tokenize(text.lower())
        for piece in alnum_pieces(token)
        if piece not in stop_words
    ]
    if stem:
        return [_stem(token) for token in tokens]
    return tokens
