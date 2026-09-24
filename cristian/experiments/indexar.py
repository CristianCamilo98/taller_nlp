"""Construye el índice FAISS del modelo de embeddings en SETTINGS.

Los vectores del corpus se generan **una vez** y se guardan en
``cristian/experiments/artifacts/indices/``. ``buscar()`` solo embebe la query.

Uso (desde la raíz del repo)::

    python -m cristian.experiments.indexar
    python -m cristian.experiments.indexar --force

El modelo lo eliges en ``config.AgentSettings.embedding_model``.
BGE local no se regenera: usa el ``corpus.faiss`` docente.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cristian.experiments.config import (
    SETTINGS,
    embedding_index_dir,
    embedding_index_path,
    embedding_manifest_path,
    embedding_model_id,
    get_dataset_paths,
    is_openrouter_embedding,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indexar(*, force: bool = False) -> Path:
    """Lee ``chunks_meta.parquet``, embebe y escribe FAISS + manifiesto."""
    if not is_openrouter_embedding():
        paths = get_dataset_paths()
        print(
            f"{SETTINGS.embedding_model} es local: no se regenera. "
            f"Usa el índice docente {paths.faiss_index}"
        )
        return paths.faiss_index

    from dotenv import load_dotenv

    from cristian.experiments.config import env_path

    load_dotenv(env_path())

    import faiss
    import pandas as pd

    from cristian.experiments.embeddings import crear_encoder

    paths = get_dataset_paths()
    out_path = embedding_index_path()
    manifest_path = embedding_manifest_path()
    if out_path.is_file() and not force:
        raise FileExistsError(
            f"Ya existe {out_path}. Pasa --force para regenerarlo, "
            "o cambia SETTINGS.embedding_model."
        )

    meta = pd.read_parquet(paths.chunks_meta)
    texts = [str(text) for text in meta["texto"].tolist()]
    model_id = embedding_model_id()
    print(
        f"Indexando {len(texts)} chunks con {SETTINGS.embedding_model} "
        f"(batch={SETTINGS.embedding_batch_size})…"
    )
    encoder = crear_encoder()
    matrix = encoder.encode_documents(texts, progress=True)
    if len(matrix) != len(meta):
        raise RuntimeError(
            f"Se obtuvieron {len(matrix)} vectores para {len(meta)} chunks"
        )

    index = faiss.IndexFlatIP(int(matrix.shape[1]))
    index.add(matrix)
    embedding_index_dir().mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    manifest = {
        "model": model_id,
        "embedding_model": SETTINGS.embedding_model,
        "dimension": int(matrix.shape[1]),
        "ntotal": int(index.ntotal),
        "index_type": "IndexFlatIP",
        "normalized": True,
        "input_type_documents": "search_document",
        "input_type_queries": "search_query",
        "chunks_jsonl_sha256": _sha256(paths.chunks),
        "chunks_meta_sha256": _sha256(paths.chunks_meta),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Guardado {out_path} ({index.ntotal} vectores, dim={index.d})\n"
        f"Manifiesto {manifest_path}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera el FAISS del embedding_model de SETTINGS"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe el índice si ya existe",
    )
    args = parser.parse_args()
    indexar(force=args.force)


if __name__ == "__main__":
    main()
