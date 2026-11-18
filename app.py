"""Servicio web de búsqueda sobre NFCorpus (TF-IDF y BM25)."""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.data import Doc, load_corpus
from src.index import InvertedIndex
from src.retrievers import BM25Retriever, Retriever, TfidfRetriever

CORPUS_NAME = "nfcorpus"
STEM = True
SNIPPET_CHARS = 250
MAX_TOP_K = 1000


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    model: Literal["bm25", "tfidf"]
    top_k: int = Field(default=10, gt=0, le=MAX_TOP_K)


class SearchHit(BaseModel):
    rank: int
    document_id: str
    score: float
    title: str
    snippet: str


class SearchResponse(BaseModel):
    model: Literal["bm25", "tfidf"]
    results: list[SearchHit]


class HealthResponse(BaseModel):
    status: str
    corpus: str
    documents: int
    models: list[str]


def make_snippet(doc: Doc) -> str:
    text = re.sub(r"\s+", " ", doc.text or doc.title).strip()
    return text[:SNIPPET_CHARS]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se indexa una sola vez al iniciar; son ~3.600 documentos
    corpus = load_corpus()
    index = InvertedIndex.build(corpus, stem=STEM)
    app.state.corpus = corpus
    app.state.retrievers = {
        "tfidf": TfidfRetriever(index, stem=STEM),
        "bm25": BM25Retriever(index, stem=STEM),
    }
    yield


app = FastAPI(title="NFCorpus Search", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        corpus=CORPUS_NAME,
        documents=len(request.app.state.corpus),
        models=["tfidf", "bm25"],
    )


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request) -> SearchResponse:
    retriever: Retriever = request.app.state.retrievers[body.model]
    corpus: dict[str, Doc] = request.app.state.corpus

    results = []
    for rank, hit in enumerate(retriever.search(body.query, k=body.top_k), start=1):
        doc = corpus[hit.doc_id]
        results.append(
            SearchHit(
                rank=rank,
                document_id=hit.doc_id,
                score=round(hit.score, 6),
                title=doc.title,
                snippet=make_snippet(doc),
            )
        )
    return SearchResponse(model=body.model, results=results)
