"""Retrieval denso común: BGE baseline, Qwen3 o Gemini Embedding 2."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common.config import DatasetPaths, get_dataset_paths
from common.retrieval.profiles import DenseRetrievalProfile, get_embedding_profile


CHUNKS_SHA256 = "388ff3671742c2248e8f5cb1c75afbc786edfa0dc6f72a62d1fadf2310ec82b2"
CHUNKS_META_SHA256 = "fbd22360e517e5da250b18f0c95dbc9ad704b41902cedb1154747fde0be16fd9"


class DenseRetrievalConfigurationError(RuntimeError):
    """El bundle de retrieval no coincide con el perfil seleccionado."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_qwen_manifest(
    profile: DenseRetrievalProfile, paths: DatasetPaths
) -> dict[str, Any]:
    """Valida manifest y archivos antes de cargar FAISS o Qwen."""
    if paths.index_manifest is None:
        raise DenseRetrievalConfigurationError("Qwen exige index_manifest.json")
    try:
        manifest = json.loads(paths.index_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseRetrievalConfigurationError(
            f"Manifest Qwen ausente o inválido: {paths.index_manifest}"
        ) from exc

    expected = {
        "schema_version": 1,
        "profile": "qwen3-06b",
        "model_name": profile.model_name,
        "model_revision": profile.model_revision,
        "embedding_dimension": 1024,
        "pooling": "lasttoken",
        "include_prompt": True,
        "normalize": True,
        "document_format": "raw",
        "query_template": profile.query_template,
        "index_type": "IndexFlatIP",
        "ntotal": 1749,
        "chunks_sha256": CHUNKS_SHA256,
        "chunks_meta_sha256": CHUNKS_META_SHA256,
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise DenseRetrievalConfigurationError(
                f"Manifest Qwen incompatible en {field}: "
                f"{manifest.get(field)!r} != {expected_value!r}"
            )

    faiss_sha256 = manifest.get("faiss_sha256")
    if (
        not isinstance(faiss_sha256, str)
        or len(faiss_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in faiss_sha256)
    ):
        raise DenseRetrievalConfigurationError(
            "Manifest Qwen incompatible en faiss_sha256"
        )

    files = {
        "faiss_sha256": paths.faiss_index,
        "chunks_sha256": paths.chunks,
        "chunks_meta_sha256": paths.chunks_meta,
    }
    for field, path in files.items():
        actual = _sha256(path)
        if actual.lower() != manifest[field].lower():
            raise DenseRetrievalConfigurationError(
                f"Hash incompatible para {field}: {actual} != {manifest[field]}"
            )
    return manifest


def _validate_gemini_manifest(
    profile: DenseRetrievalProfile, paths: DatasetPaths
) -> dict[str, Any]:
    """Valida el bundle histórico Gemini antes de cualquier llamada de red."""
    if paths.index_manifest is None:
        raise DenseRetrievalConfigurationError(
            "Gemini exige index_manifest.json"
        )
    try:
        manifest = json.loads(paths.index_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseRetrievalConfigurationError(
            f"Manifest Gemini ausente o inválido: {paths.index_manifest}"
        ) from exc

    expected = {
        "schema_version": 1,
        "profile": "gemini-embedding-2",
        "provider": "openrouter",
        "model_name": profile.model_name,
        "model_revision": None,
        "model_revision_policy": "provider_managed",
        "embedding_dimension": 3072,
        "normalize": True,
        "document_format": "raw",
        "input_type_documents_requested": "search_document",
        "input_type_queries_requested": "search_query",
        "input_type_historical_effective": "unknown",
        "index_type": "IndexFlatIP",
        "ntotal": 1749,
        "chunks_sha256": CHUNKS_SHA256,
        "chunks_meta_sha256": CHUNKS_META_SHA256,
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise DenseRetrievalConfigurationError(
                f"Manifest Gemini incompatible en {field}: "
                f"{manifest.get(field)!r} != {expected_value!r}"
            )
    faiss_sha256 = manifest.get("faiss_sha256")
    if (
        not isinstance(faiss_sha256, str)
        or len(faiss_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in faiss_sha256)
    ):
        raise DenseRetrievalConfigurationError(
            "Manifest Gemini incompatible en faiss_sha256"
        )
    for field, path in {
        "faiss_sha256": paths.faiss_index,
        "chunks_sha256": paths.chunks,
        "chunks_meta_sha256": paths.chunks_meta,
    }.items():
        actual = _sha256(path)
        if actual.lower() != manifest[field].lower():
            raise DenseRetrievalConfigurationError(
                f"Hash incompatible para {field}: {actual} != {manifest[field]}"
            )
    return manifest


def _validate_index(
    profile: DenseRetrievalProfile,
    index: Any,
    metadata: Any,
    manifest: dict[str, Any] | None,
) -> None:
    if type(index).__name__ != profile.index_type:
        raise DenseRetrievalConfigurationError(
            f"Tipo de índice {type(index).__name__!r}; "
            f"esperado {profile.index_type!r}"
        )
    if int(getattr(index, "d", -1)) != profile.dimension:
        raise DenseRetrievalConfigurationError(
            f"Dimensión de índice {getattr(index, 'd', None)}; "
            f"esperada {profile.dimension}"
        )
    if int(getattr(index, "ntotal", -1)) != len(metadata):
        raise DenseRetrievalConfigurationError(
            f"Índice y metadata desalineados: "
            f"{getattr(index, 'ntotal', None)} != {len(metadata)}"
        )
    if manifest is not None and index.ntotal != manifest["ntotal"]:
        raise DenseRetrievalConfigurationError(
            f"ntotal incompatible: {index.ntotal} != {manifest['ntotal']}"
        )


def _validate_encoder(profile: DenseRetrievalProfile, encoder: Any) -> None:
    dimension = encoder.get_sentence_embedding_dimension()
    if int(dimension) != profile.dimension:
        raise DenseRetrievalConfigurationError(
            f"Dimensión del encoder {dimension}; esperada {profile.dimension}"
        )
    if profile.name != "qwen3-06b":
        return
    try:
        pooling = encoder[1].get_config_dict()
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise DenseRetrievalConfigurationError(
            "No se puede verificar el pooling efectivo de Qwen"
        ) from exc
    if pooling.get("pooling_mode") != "lasttoken":
        raise DenseRetrievalConfigurationError(
            "Qwen debe utilizar pooling_mode='lasttoken'"
        )
    if pooling.get("include_prompt") is not True:
        raise DenseRetrievalConfigurationError(
            "Qwen debe utilizar include_prompt=True"
        )


def _create_encoder(
    profile: DenseRetrievalProfile,
    encoder_factory: Callable[..., Any] | None = None,
) -> Any:
    if profile.name == "gemini-embedding-2":
        from common.retrieval.openrouter_embeddings import (
            OpenRouterEmbeddingEncoder,
        )

        return OpenRouterEmbeddingEncoder(profile.model_name, profile.dimension)
    if encoder_factory is None:
        from sentence_transformers import SentenceTransformer

        encoder_factory = SentenceTransformer
    if profile.name == "qwen3-06b":
        return encoder_factory(
            profile.model_name,
            revision=profile.model_revision,
            local_files_only=True,
        )
    return encoder_factory(profile.model_name)


@functools.lru_cache(maxsize=3)
def _load_profile_resources(profile_name: str):
    profile = get_embedding_profile(profile_name)
    paths = get_dataset_paths(profile.name)
    if profile.name == "qwen3-06b":
        manifest = _validate_qwen_manifest(profile, paths)
    elif profile.name == "gemini-embedding-2":
        manifest = _validate_gemini_manifest(profile, paths)
    else:
        manifest = None

    import faiss
    import pandas as pd

    index = faiss.read_index(str(paths.faiss_index))
    metadata = pd.read_parquet(paths.chunks_meta)
    _validate_index(profile, index, metadata, manifest)

    encoder = _create_encoder(profile)
    _validate_encoder(profile, encoder)
    return index, metadata, encoder, profile


def _indice(profile_name: str | None = None):
    profile = get_embedding_profile(profile_name)
    return _load_profile_resources(profile.name)


def clear_dense_retrieval_cache() -> None:
    _load_profile_resources.cache_clear()


def get_last_query_embedding_metadata() -> dict[str, Any] | None:
    """Devuelve provenance de la última query OpenRouter, si aplica."""
    _index, _metadata, encoder, profile = _indice()
    if profile.name != "gemini-embedding-2":
        return None
    value = getattr(encoder, "last_request_metadata", None)
    return dict(value) if value is not None else None


def buscar(query: str, ticker: str | None = None,
           fiscal_year: int | None = None, item: str | None = None,
           k: int = 5, *, profile_name: str | None = None) -> list[dict]:
    """Ranking global denso seguido de filtros de metadata."""
    index, metadata, encoder, profile = _indice(profile_name)
    vector = encoder.encode(
        [profile.format_query(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    if vector.shape != (1, profile.dimension) or not np.isfinite(vector).all():
        raise DenseRetrievalConfigurationError("Embedding de query inválido")
    norm = float(np.linalg.norm(vector[0]))
    tolerance = 5e-3 if profile.name == "qwen3-06b" else 1e-4
    if not np.isclose(norm, 1.0, rtol=tolerance, atol=1e-5):
        raise DenseRetrievalConfigurationError(
            f"El embedding de query no está normalizado en L2: {norm}"
        )
    if profile.name in {"qwen3-06b", "gemini-embedding-2"}:
        # Reafirma L2 después del cast a FP32 para los perfiles no baseline.
        vector /= norm

    scores, positions = index.search(vector, index.ntotal)
    results = []
    for score, position in zip(scores[0], positions[0]):
        row = metadata.iloc[int(position)]
        if ticker and row["ticker"] != ticker:
            continue
        if fiscal_year and int(row["fiscal_year"]) != int(fiscal_year):
            continue
        if item and row["item"] != item:
            continue
        results.append({
            "chunk_id": row["chunk_id"],
            "ticker": row["ticker"],
            "fiscal_year": int(row["fiscal_year"]),
            "item": row["item"],
            "texto": row["texto"],
            "n_tokens": int(row["n_tokens"]),
            "contiene_tabla": bool(row["contiene_tabla"]),
            "puntuacion": round(float(score), 4),
        })
        if len(results) >= k:
            break
    return results


def formatear_fragmentos(fragmentos: list[dict]) -> str:
    """Formatea resultados exactamente como el baseline común."""
    if not fragmentos:
        return ("Sin resultados para esa consulta con esos filtros. "
                "Prueba a quitar algún filtro o a reformular la búsqueda.")
    partes = []
    for fragmento in fragmentos:
        partes.append(
            f"[{fragmento['chunk_id']}] {fragmento['ticker']} "
            f"FY{fragmento['fiscal_year']} Item {fragmento['item']} "
            f"(similitud {fragmento['puntuacion']:.3f})\n"
            f"{fragmento['texto']}"
        )
    return "\n\n---\n\n".join(partes)
