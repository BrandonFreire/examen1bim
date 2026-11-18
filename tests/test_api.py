"""Pruebas rápidas de la API con TestClient (el `with` ejecuta el lifespan)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import app
from src.data import load_corpus, load_qrels, load_queries

QUERY = "statins breast cancer survival"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def corpus_ids() -> set[str]:
    return set(load_corpus())


def search(client: TestClient, model: str, query: str = QUERY, top_k: int = 10):
    return client.post("/search", json={"query": query, "model": model, "top_k": top_k})


def test_health(client: TestClient, corpus_ids: set[str]) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "ok",
        "corpus": "nfcorpus",
        "documents": len(corpus_ids),
        "models": ["tfidf", "bm25"],
    }


@pytest.mark.parametrize("model", ["bm25", "tfidf"])
def test_search_format_order_and_ids(
    client: TestClient, corpus_ids: set[str], model: str
) -> None:
    response = search(client, model, top_k=10)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model", "results"}
    assert body["model"] == model

    results = body["results"]
    assert 0 < len(results) <= 10
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
    for r in results:
        assert set(r) == {"rank", "document_id", "score", "title", "snippet"}
        assert isinstance(r["document_id"], str) and r["document_id"] in corpus_ids
        assert len(r["snippet"]) <= 250

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    ids = [r["document_id"] for r in results]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("model", ["bm25", "tfidf"])
def test_top_k_is_respected(client: TestClient, model: str) -> None:
    assert len(search(client, model, top_k=3).json()["results"]) == 3
    assert len(search(client, model, top_k=50).json()["results"]) == 50


@pytest.mark.parametrize("model", ["bm25", "tfidf"])
def test_test_query_finds_relevant_documents(client: TestClient, model: str) -> None:
    # Con consultas reales de test debe aparecer al menos un documento relevante en el top-10
    queries, qrels = load_queries(), load_qrels("test")
    hits = 0
    for qid in sorted(qrels)[:20]:
        ids = {r["document_id"] for r in search(client, model, queries[qid]).json()["results"]}
        hits += bool(ids & set(qrels[qid]))
    assert hits >= 10


def test_unknown_terms_return_empty_results(client: TestClient) -> None:
    response = search(client, "bm25", "zzzqqqxxx")
    assert response.status_code == 200
    assert response.json() == {"model": "bm25", "results": []}


@pytest.mark.parametrize(
    "payload",
    [
        {"query": QUERY, "model": "dense", "top_k": 10},
        {"query": QUERY, "model": "bm25", "top_k": 0},
        {"query": QUERY, "model": "bm25", "top_k": -5},
        {"query": "", "model": "bm25", "top_k": 10},
        {"model": "bm25", "top_k": 10},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, payload: dict) -> None:
    assert client.post("/search", json=payload).status_code in (400, 422)
