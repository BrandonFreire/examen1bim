"""Carga de NFCorpus (documentos, consultas y qrels) desde Hugging Face."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

CORPUS_REPO = "BeIR/nfcorpus"
QRELS_REPO = "BeIR/nfcorpus-qrels"
# Revisiones fijas para que los datos sean siempre los mismos
CORPUS_REVISION = "b5026a0e96e8a7ac4f95f482a596389289d46269"
QRELS_REVISION = "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65"
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class Doc:
    title: str
    text: str


def _download(repo: str, filename: str, revision: str) -> Path:
    return Path(
        hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    )


def load_corpus() -> dict[str, Doc]:
    """document_id (str, tal cual) -> Doc."""
    path = _download(CORPUS_REPO, "corpus/corpus-00000-of-00001.parquet", CORPUS_REVISION)
    rows = pq.read_table(path).to_pylist()
    return {
        str(row["_id"]): Doc(title=row["title"] or "", text=row["text"] or "")
        for row in rows
    }


def load_queries() -> dict[str, str]:
    """query_id (str, tal cual) -> texto de la consulta."""
    path = _download(CORPUS_REPO, "queries/queries-00000-of-00001.parquet", CORPUS_REVISION)
    rows = pq.read_table(path).to_pylist()
    return {str(row["_id"]): row["text"] for row in rows}


def load_qrels(split: str) -> dict[str, dict[str, int]]:
    """query_id -> {document_id: score}. Los IDs se dejan como str."""
    if split not in SPLITS:
        raise ValueError(f"split debe ser uno de {SPLITS}, se recibió {split!r}")

    path = _download(QRELS_REPO, f"{split}.tsv", QRELS_REVISION)
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


def download_all() -> None:
    """Deja todos los archivos en la caché local (útil al construir la imagen Docker)."""
    load_corpus()
    load_queries()
    for split in SPLITS:
        load_qrels(split)
