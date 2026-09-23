"""Genera el índice FAISS con Qwen3-Embedding-0.6B.

Reutiliza chunks.jsonl y chunks_meta.parquet. Guarda el índice en
dataset/indice_faiss_qwen3/corpus.faiss para no tocar el de BGE-small.

Parámetros idénticos a dani/experiments/config.py e3_qwen3.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Config Qwen3 (copiada de dani/experiments/config.py)
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QUERY_PREFIX = (
    "Instruct: Given a financial question, retrieve relevant passages "
    "from SEC 10-K filings that answer the question.\nQuery:"
)
EXPECTED_DIM = 1024
NORMALIZE = True


def main():
    # Rutas
    chunks_path = RAIZ / "dataset" / "corpus_miax_2026" / "chunks.jsonl"
    meta_path = RAIZ / "dataset" / "indice_faiss" / "chunks_meta.parquet"
    out_dir = RAIZ / "dataset" / "indice_faiss_qwen3"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.is_file():
        raise FileNotFoundError(f"No existe {chunks_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"No existe {meta_path}")

    # Cargar chunks
    print("Cargando chunks.jsonl...")
    chunks = [json.loads(line) for line in open(chunks_path, encoding="utf-8") if line.strip()]
    meta = pd.read_parquet(meta_path)
    print(f"  {len(chunks)} chunks, {len(meta)} metadatos")

    if len(chunks) != len(meta):
        raise RuntimeError(
            f"Desalineación: {len(chunks)} chunks vs {len(meta)} metadatos"
        )

    # Verificar orden por chunk_id
    chunks_ordenados = sorted(chunks, key=lambda c: c["chunk_id"])
    meta_ordenados = meta.sort_values("chunk_id").reset_index(drop=True)
    if [c["chunk_id"] for c in chunks_ordenados] != list(meta_ordenados["chunk_id"]):
        raise RuntimeError("Los chunk_id no coinciden entre chunks y meta")

    # Cargar modelo
    print(f"Cargando {MODEL_NAME} (revision {MODEL_REVISION[:10]}...)...")
    t0 = time.time()
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",  # ajusta si tienes GPU
    )
    print(f"  Modelo cargado en {time.time() - t0:.1f}s")

    # Verificar dimensión
    dim = model.get_sentence_embedding_dimension()
    print(f"  Dimensión: {dim} (esperada: {EXPECTED_DIM})")
    if dim != EXPECTED_DIM:
        raise RuntimeError(f"Dimensión incorrecta: {dim}")

    # Codificar documentos (sin prefijo, según dani/experiments/config.py)
    print("Codificando 1749 documentos...")
    textos = [c["texto"] for c in chunks_ordenados]
    t0 = time.time()
    embeddings = model.encode(
        textos,
        normalize_embeddings=NORMALIZE,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=True,
    ).astype("float32")
    print(f"  Codificados en {time.time() - t0:.1f}s, shape: {embeddings.shape}")

    # Construir índice FAISS
    print("Construyendo índice FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(EXPECTED_DIM)
    index.add(embeddings)
    print(f"  {index.ntotal} vectores indexados")

    # Guardar
    faiss_path = out_dir / "corpus.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"  Índice guardado en {faiss_path}")

    # Copiar metadatos
    meta_out = out_dir / "chunks_meta.parquet"
    meta_ordenados.to_parquet(meta_out, index=False)
    print(f"  Metadatos guardados en {meta_out}")

    # Verificación de round-trip
    print("Verificando round-trip...")
    idx2 = faiss.read_index(str(faiss_path))
    meta2 = pd.read_parquet(meta_out)
    if idx2.ntotal != len(meta2):
        raise RuntimeError("Desalineación tras guardar")
    print(f"  OK: {idx2.ntotal} vectores, {len(meta2)} filas")

    # Guardar un README con la config
    readme = out_dir / "README.md"
    readme.write_text(f"""# Índice Qwen3-Embedding-0.6B

- Modelo: {MODEL_NAME}
- Revision: {MODEL_REVISION}
- Query prefix: `{QUERY_PREFIX}`
- Dimension: {EXPECTED_DIM}
- Normalize: {NORMALIZE}
- Chunks: {len(chunks)}
- Construido: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Uso

`MIAX_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B`
`MIAX_INDEX_DIRNAME=indice_faiss_qwen3`
""", encoding="utf-8")
    print(f"  README guardado en {readme}")

    print("\nDONE")


if __name__ == "__main__":
    main()