"""Cliente mínimo para embeddings OpenRouter usados por common."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
QUERY_INPUT_TYPE = "search_query"
_LOGGER = logging.getLogger(__name__)


class OpenRouterEmbeddingError(RuntimeError):
    """La petición de embeddings falló o devolvió un payload inválido."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def _api_key() -> str:
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterEmbeddingError("Falta OPENROUTER_API_KEY")
    return key


def _post_openrouter(payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        OPENROUTER_EMBEDDINGS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/miax-taller-nlp",
            "X-Title": "miax-common-runtime",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = OpenRouterEmbeddingError(
                f"OpenRouter embeddings HTTP {exc.code}: {detail[:500]}",
                status=exc.code,
            )
            if exc.code in {429, 502, 503} and attempt < 2:
                time.sleep(5 * (2**attempt))
                continue
            raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = OpenRouterEmbeddingError(
                f"OpenRouter embeddings network error: {exc.reason}"
            )
            if attempt < 2:
                time.sleep(5 * (2**attempt))
                continue
            raise last_error from exc
    raise last_error or OpenRouterEmbeddingError("Unreachable embedding loop")


class OpenRouterEmbeddingEncoder:
    """Encoder compatible con el uso de ``SentenceTransformer.encode``."""

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension
        self.last_request_metadata: dict[str, Any] | None = None

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **_kwargs) -> np.ndarray:
        values = list(texts)
        if not values:
            raise OpenRouterEmbeddingError("No hay textos para embeber")
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": values,
            "input_type": QUERY_INPUT_TYPE,
        }
        mode = "requested"
        try:
            response = _post_openrouter(payload)
        except OpenRouterEmbeddingError as exc:
            if exc.status != 400:
                raise
            mode = "fallback_without_input_type"
            payload.pop("input_type")
            message = (
                "OpenRouter rechazó input_type='search_query'; "
                "se reintenta sin input_type"
            )
            _LOGGER.warning(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            response = _post_openrouter(payload)

        items = response.get("data")
        if not isinstance(items, list) or len(items) != len(values):
            raise OpenRouterEmbeddingError("Respuesta de embeddings inválida")
        ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
        matrix = np.asarray(
            [item.get("embedding") for item in ordered], dtype=np.float32
        )
        if matrix.shape != (len(values), self.dimension):
            raise OpenRouterEmbeddingError(
                f"Shape de embeddings {matrix.shape}; "
                f"esperada ({len(values)}, {self.dimension})"
            )
        if not np.isfinite(matrix).all():
            raise OpenRouterEmbeddingError("Embedding con NaN o infinito")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise OpenRouterEmbeddingError("Embedding con norma cero")
        matrix = np.asarray(matrix / norms, dtype=np.float32)
        self.last_request_metadata = {
            "provider": "openrouter",
            "model_name": self.model_name,
            "input_type_requested": QUERY_INPUT_TYPE,
            "input_type_mode": mode,
            "usage": response.get("usage"),
        }
        return matrix
