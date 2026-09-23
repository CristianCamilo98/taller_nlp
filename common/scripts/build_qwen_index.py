"""Construye el bundle FAISS Qwen desde los 1.749 chunks oficiales."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import get_dataset_paths
from common.retrieval.dense_baseline import (
    CHUNKS_META_SHA256,
    CHUNKS_SHA256,
    _create_encoder,
    _validate_encoder,
)
from common.retrieval.profiles import get_embedding_profile

EXPECTED_CHUNKS = 1749
BATCH_SIZE = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_documents(chunks_path: Path, metadata: pd.DataFrame) -> list[str]:
    records = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != EXPECTED_CHUNKS or len(metadata) != EXPECTED_CHUNKS:
        raise RuntimeError(
            f"Se esperaban {EXPECTED_CHUNKS} chunks; "
            f"JSONL={len(records)}, metadata={len(metadata)}"
        )
    chunk_ids = [record.get("chunk_id") for record in records]
    if chunk_ids != metadata["chunk_id"].tolist():
        raise RuntimeError("El orden de chunks.jsonl y chunks_meta no coincide")
    documents = [record.get("texto") for record in records]
    if any(not isinstance(text, str) or not text for text in documents):
        raise RuntimeError("chunks.jsonl contiene documentos vacíos o inválidos")
    if documents != metadata["texto"].tolist():
        raise RuntimeError("El texto de chunks.jsonl y chunks_meta no coincide")
    return documents


def _float32_l2(vectors: object) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (EXPECTED_CHUNKS, 1024):
        raise RuntimeError(
            f"Shape de embeddings {matrix.shape}; esperada ({EXPECTED_CHUNKS}, 1024)"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("Los embeddings contienen NaN o infinito")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Qwen produjo un embedding con norma cero")
    matrix = np.asarray(matrix / norms, dtype=np.float32)
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise RuntimeError("No se pudo normalizar los embeddings en L2")
    return matrix


def _manifest(profile, faiss_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "profile": profile.name,
        "model_name": profile.model_name,
        "model_revision": profile.model_revision,
        "embedding_dimension": profile.dimension,
        "pooling": "lasttoken",
        "include_prompt": True,
        "normalize": True,
        "document_format": "raw",
        "query_template": profile.query_template,
        "index_type": profile.index_type,
        "ntotal": EXPECTED_CHUNKS,
        "faiss_sha256": faiss_sha256,
        "chunks_sha256": CHUNKS_SHA256,
        "chunks_meta_sha256": CHUNKS_META_SHA256,
    }


def build() -> Path:
    """Crea el bundle una sola vez; nunca sobrescribe uno existente."""
    source = get_dataset_paths("bge-small-baseline")
    target = source.dataset_dir / "indice_faiss_qwen3_06b"
    if target.exists():
        raise RuntimeError(f"El destino ya existe; no se sobrescribe: {target}")
    if _sha256(source.chunks) != CHUNKS_SHA256:
        raise RuntimeError("SHA-256 inesperado para chunks.jsonl")
    if _sha256(source.chunks_meta) != CHUNKS_META_SHA256:
        raise RuntimeError("SHA-256 inesperado para chunks_meta.parquet")

    metadata = pd.read_parquet(source.chunks_meta)
    documents = _read_documents(source.chunks, metadata)
    profile = get_embedding_profile("qwen3-06b")
    encoder = _create_encoder(profile)
    _validate_encoder(profile, encoder)
    vectors = encoder.encode(
        documents,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    matrix = _float32_l2(vectors)

    import faiss

    index = faiss.IndexFlatIP(profile.dimension)
    index.add(matrix)
    if type(index).__name__ != "IndexFlatIP" or index.d != 1024:
        raise RuntimeError("El índice generado no es IndexFlatIP de dimensión 1024")
    if index.ntotal != EXPECTED_CHUNKS:
        raise RuntimeError(f"ntotal inesperado: {index.ntotal}")

    temporary = Path(tempfile.mkdtemp(prefix=".qwen-index-", dir=source.dataset_dir))
    try:
        faiss_path = temporary / "corpus.faiss"
        metadata_path = temporary / "chunks_meta.parquet"
        manifest_path = temporary / "index_manifest.json"
        faiss.write_index(index, str(faiss_path))
        shutil.copyfile(source.chunks_meta, metadata_path)
        faiss_sha256 = _sha256(faiss_path)
        manifest_path.write_text(
            json.dumps(_manifest(profile, faiss_sha256), indent=2) + "\n",
            encoding="utf-8",
        )
        written = faiss.read_index(str(faiss_path))
        if type(written).__name__ != "IndexFlatIP" or written.d != 1024:
            raise RuntimeError("La verificación del FAISS escrito falló")
        if written.ntotal != EXPECTED_CHUNKS:
            raise RuntimeError("La verificación de ntotal escrito falló")
        if _sha256(metadata_path) != CHUNKS_META_SHA256:
            raise RuntimeError("La copia de chunks_meta cambió sus bytes")
        temporary.replace(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    print(json.dumps({"bundle": str(target), "faiss_sha256": faiss_sha256}))
    return target


if __name__ == "__main__":
    build()
