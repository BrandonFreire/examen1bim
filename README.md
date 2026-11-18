# Buscador NFCorpus (TF-IDF y BM25)

Servicio web de recuperación de información sobre **NFCorpus** (`BeIR/nfcorpus` y `BeIR/nfcorpus-qrels`). Indexa los 3.633 documentos en memoria con un índice invertido propio y ofrece dos rankings: **TF-IDF con similitud coseno** y **BM25** (`k1=1.5`, `b=0.75`). Hecho con FastAPI y NLTK, sin librerías externas de búsqueda.

El análisis y las métricas (`P@10`, `R@10`, `NDCG@10`) están en `examen1bim.ipynb`.

## Endpoints

- `GET /health` → `{"status": "ok", "corpus": "nfcorpus", "documents": 3633, "models": ["tfidf", "bm25"]}`
- `POST /search` con `{"query": "statins and breast cancer", "model": "bm25", "top_k": 10}`
  - `model`: `bm25` o `tfidf`. Un modelo inválido o `top_k <= 0` devuelve error 422.
  - Respuesta: `{"model": ..., "results": [{"rank", "document_id", "score", "title", "snippet"}]}`

La documentación interactiva queda en `/docs`.

## Ejecutar en local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest httpx
pytest -q
uvicorn app:app --port 10000
```

Probar:

```bash
curl http://127.0.0.1:10000/health
curl -X POST http://127.0.0.1:10000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "statins and breast cancer", "model": "bm25", "top_k": 5}'
```

La primera vez descarga el corpus de Hugging Face (unos 3 MB).

## Desplegar en Render (Docker)

1. Sube el proyecto a GitHub.
2. En [render.com](https://render.com): **New → Web Service** y elige el repositorio.
3. Configura:
   - **Language:** `Docker` (usa el `Dockerfile` de la raíz)
   - **Instance Type:** `Free`
   - Sin variables de entorno: Render define `PORT` solo.
   - **Health Check Path** (opcional): `/health`
4. Pulsa **Create Web Service**. El build descarga el corpus y los recursos de NLTK, así que el arranque es rápido.
5. Cuando diga *Live*, la URL será `https://<nombre>.onrender.com`. Pruébala en Postman:
   - `GET https://<nombre>.onrender.com/health`
   - `POST https://<nombre>.onrender.com/search` con Body → raw → JSON.

> En el plan gratuito el servicio se duerme tras unos 15 minutos sin uso; la primera petición después puede tardar. Abre `/health` antes de una demostración.

## Estructura

```
app.py                 # API FastAPI
src/data.py            # carga de documentos, consultas y qrels
src/preprocessing.py   # tokenización, stopwords y stemming
src/index.py           # índice invertido
src/retrievers.py      # TF-IDF y BM25
src/evaluation.py      # P@10, R@10, NDCG@10
tests/test_api.py      # pruebas de la API
Dockerfile
requirements.txt
examen1bim.ipynb
```
