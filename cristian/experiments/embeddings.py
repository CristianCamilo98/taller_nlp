"""Codificadores de embeddings: BGE local o OpenRouter.

La geometría es la misma en ambos backends: vectores L2-normalizados para
usarlos con ``IndexFlatIP`` (similitud coseno). Lo que cambia es *quién*
produce el vector:

- ``BAAI/bge-small-en-v1.5`` → SentenceTransformer (prefijo solo en la query).
- ``openrouter:...`` → POST /api/v1/embeddings (``input_type`` query/document).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol

import numpy as np

from cristian.experiments.config import (
    SETTINGS,
    embedding_model_id,
    env_path,
    is_openrouter_embedding,
)

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
INPUT_TYPE_QUERY = "search_query"
INPUT_TYPE_DOCUMENT = "search_document"


class EmbeddingEncoder(Protocol):
    def encode_query(self, query: str) -> np.ndarray: ...
    def encode_documents(self, texts: list[str], *, progress: bool = False) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.clip(norms, 1e-12, None)).astype("float32", copy=False)


def _openrouter_api_key() -> str:
    from dotenv import load_dotenv

    load_dotenv(env_path())
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            f"Falta OPENROUTER_API_KEY en {env_path()}. "
            "Hace falta para embeddings OpenRouter."
        )
    return key


class OpenRouterEmbeddingError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _post_openrouter(payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {_openrouter_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/miax-taller-nlp",
        "X-Title": "cristian-experiments",
    }
    attempts = SETTINGS.rate_limit_attempts
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            OPENROUTER_EMBEDDINGS_URL,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            last_error = OpenRouterEmbeddingError(
                f"OpenRouter embeddings HTTP {error.code}: {detail[:500]}",
                status=error.code,
            )
            if error.code in {429, 502, 503} and attempt < attempts - 1:
                wait = SETTINGS.rate_limit_initial_backoff_s * (2 ** attempt)
                retry_after = error.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
                time.sleep(wait)
                continue
            raise last_error from error
        except urllib.error.URLError as error:
            last_error = OpenRouterEmbeddingError(
                f"OpenRouter embeddings red: {error.reason}"
            )
            if attempt < attempts - 1:
                time.sleep(SETTINGS.rate_limit_initial_backoff_s * (2 ** attempt))
                continue
            raise last_error from error
    raise last_error or OpenRouterEmbeddingError("bucle de embeddings inalcanzable")


def embed_openrouter(
    texts: list[str],
    *,
    model: str | None = None,
    input_type: str | None = None,
    progress: bool = False,
) -> np.ndarray:
    """Embedde ``texts`` con el modelo OpenRouter elegido en SETTINGS."""
    if not texts:
        raise ValueError("no hay textos para embeber")
    model_id = embedding_model_id(model)
    batch_size = max(1, SETTINGS.embedding_batch_size)
    rows: list[list[float]] = []
    total = len(texts)
    use_input_type = input_type
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        payload: dict = {"model": model_id, "input": batch}
        if use_input_type:
            payload["input_type"] = use_input_type
        try:
            data = _post_openrouter(payload)
        except OpenRouterEmbeddingError as error:
            # Algunos modelos no aceptan input_type; reintentar el batch sin él.
            if use_input_type and error.status == 400:
                use_input_type = None
                payload.pop("input_type", None)
                data = _post_openrouter(payload)
            else:
                raise
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(batch):
            got = "no-list" if not isinstance(items, list) else str(len(items))
            raise OpenRouterEmbeddingError(
                f"Respuesta inesperada de embeddings ({model_id}): "
                f"esperaba {len(batch)} vectores, recibí {got}"
            )
        ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise OpenRouterEmbeddingError(
                    f"Vector vacío o inválido en la respuesta de {model_id}"
                )
            rows.append(vector)
        if progress:
            done = min(start + len(batch), total)
            print(f"  embeddings {done}/{total}", flush=True)
    matrix = np.asarray(rows, dtype="float32")
    if matrix.ndim != 2:
        raise OpenRouterEmbeddingError("la matriz de embeddings no es 2-D")
    return _l2_normalize(matrix)


class OpenRouterEncoder:
    def __init__(self, model: str | None = None):
        self._model = model

    def encode_query(self, query: str) -> np.ndarray:
        return embed_openrouter(
            [query], model=self._model, input_type=INPUT_TYPE_QUERY
        )

    def encode_documents(self, texts: list[str], *, progress: bool = False) -> np.ndarray:
        return embed_openrouter(
            texts,
            model=self._model,
            input_type=INPUT_TYPE_DOCUMENT,
            progress=progress,
        )


class LocalBGEEncoder:
    def __init__(self, model_id: str, query_prefix: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id)
        self._query_prefix = query_prefix

    def encode_query(self, query: str) -> np.ndarray:
        return self._model.encode(
            [self._query_prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_documents(self, texts: list[str], *, progress: bool = False) -> np.ndarray:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
            batch_size=SETTINGS.embedding_batch_size,
        ).astype("float32")


def crear_encoder(model: str | None = None) -> EmbeddingEncoder:
    chosen = SETTINGS.embedding_model if model is None else model
    if is_openrouter_embedding(chosen):
        return OpenRouterEncoder(chosen)
    return LocalBGEEncoder(embedding_model_id(chosen), SETTINGS.embedding_query_prefix)
