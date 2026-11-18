---
title: NFCorpus Search
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# NFCorpus Search (TF-IDF y BM25)

Servicio web FastAPI con índice invertido propio sobre NFCorpus (`BeIR/nfcorpus`).

- `GET /health`
- `POST /search` con `{"query": "...", "model": "bm25" | "tfidf", "top_k": 10}`

Documentación interactiva en `/docs`.
