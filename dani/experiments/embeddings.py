"""Adaptadores mínimos para las familias de embeddings evaluadas."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from dani.experiments.config import ExperimentConfig

FloatMatrix = NDArray[np.float32]


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Aísla el formato específico de query/documento de cada familia."""

    def encode_query(self, text: str) -> FloatMatrix: ...

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix: ...

    def load(self) -> float: ...

    def token_lengths(self, texts: Sequence[str]) -> list[int]: ...

    def metadata(self) -> dict[str, Any]: ...


class _SentenceTransformerAdapter:
    """Mecánica compartida; cada familia define su propio formato textual."""

    def __init__(self, config: ExperimentConfig, encoder: Any | None = None):
        self.config = config
        self._encoder = encoder
        self._model_load_s = 0.0 if encoder is not None else None

    def load(self) -> float:
        """Carga solo desde caché local; nunca descarga el modelo."""
        encoder = self._ensure_encoder()
        self._validate_model_metadata(encoder)
        return float(self._model_load_s or 0.0)

    def format_query(self, text: str) -> str:
        raise NotImplementedError

    def format_documents(self, texts: Sequence[str]) -> list[str]:
        raise NotImplementedError

    def encode_query(self, text: str) -> FloatMatrix:
        vectors = self._encode([self.format_query(text)])
        if vectors.shape[0] != 1:
            raise ValueError("La codificación de una query debe producir una fila")
        return vectors

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            raise ValueError("No se puede construir un índice sin documentos")
        return self._encode(self.format_documents(texts))

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Longitudes con tokens especiales antes de cualquier truncation."""
        encoder = self._ensure_encoder()
        tokenizer = getattr(encoder, "tokenizer", None)
        if tokenizer is None:
            try:
                tokenizer = encoder[0].tokenizer
            except (AttributeError, IndexError, KeyError, TypeError) as exc:
                raise RuntimeError("El tokenizer efectivo no está disponible") from exc
        encoded = tokenizer(
            self.format_documents(texts),
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        lengths = encoded.get("length")
        if lengths is None:
            lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
        return [int(length) for length in lengths]

    def metadata(self) -> dict[str, Any]:
        encoder = self._encoder
        model_config = None
        if encoder is not None:
            try:
                model_config = encoder[0].auto_model.config
            except (AttributeError, IndexError, KeyError, TypeError):
                model_config = None
        revision = getattr(model_config, "_commit_hash", None)
        max_length = getattr(encoder, "max_seq_length", None)
        device = getattr(encoder, "device", None)
        effective_dimension = self._effective_dimension(encoder)
        return {
            "model_name": self.config.model_name,
            "model_revision": revision or self.config.model_revision,
            "dimension": effective_dimension,
            "normalize": self.config.normalize_embeddings,
            "query_formatting": f"{self.config.query_prefix}<query>",
            "document_formatting": self.config.document_format,
            "max_sequence_length": max_length,
            "requested_device": self.config.requested_device,
            "effective_device": str(device) if device is not None else None,
            "batch_size": self.config.embedding_batch_size,
            "model_load_s": self._model_load_s,
        }

    def _validate_model_metadata(self, encoder: Any) -> None:
        dimension = self._effective_dimension(encoder)
        if dimension != self.config.expected_dimension:
            raise ValueError(
                f"Dimensión efectiva {dimension}; esperada "
                f"{self.config.expected_dimension}"
            )
        max_length = getattr(encoder, "max_seq_length", None)
        if max_length != self.config.expected_max_sequence_length:
            raise ValueError(
                f"max_seq_length efectivo {max_length}; esperado "
                f"{self.config.expected_max_sequence_length}"
            )

    def _effective_dimension(self, encoder: Any | None) -> int:
        if encoder is None:
            return self.config.expected_dimension
        getter = getattr(encoder, "get_embedding_dimension", None)
        if getter is None:
            getter = getattr(encoder, "get_sentence_embedding_dimension", None)
        if getter is None:
            return self.config.expected_dimension
        dimension = getter()
        return int(dimension) if dimension is not None else self.config.expected_dimension

    def _ensure_encoder(self) -> Any:
        if self._encoder is None:
            start = perf_counter()
            from sentence_transformers import SentenceTransformer

            try:
                self._encoder = SentenceTransformer(
                    self._model_source(),
                    device=self.config.requested_device,
                    local_files_only=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"El modelo {self.config.model_name!r} no está disponible "
                    "completamente en la caché local"
                ) from exc
            self._model_load_s = perf_counter() - start
        return self._encoder

    def _model_source(self) -> str:
        return self.config.model_name

    def _encode(self, texts: Sequence[str]) -> FloatMatrix:
        encoder = self._ensure_encoder()
        vectors = encoder.encode(
            list(texts),
            batch_size=self.config.embedding_batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(vectors, dtype=np.float32)
        matrix = self._postprocess_matrix(matrix)
        self._validate(matrix, expected_rows=len(texts))
        return matrix

    def _postprocess_matrix(self, matrix: FloatMatrix) -> FloatMatrix:
        return matrix

    def _validate(self, matrix: FloatMatrix, expected_rows: int) -> None:
        expected_shape = (expected_rows, self.config.expected_dimension)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Shape de embeddings {matrix.shape}; esperado {expected_shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Los embeddings contienen NaN o infinito")
        if self.config.normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
                raise ValueError("Los embeddings no están normalizados en L2")


class BgeV15Adapter(_SentenceTransformerAdapter):
    """BGE v1.5: prefijo solo en queries y documentos sin prefijo."""

    def format_query(self, text: str) -> str:
        return f"{self.config.query_prefix}{text}"

    def format_documents(self, texts: Sequence[str]) -> list[str]:
        return list(texts)


class E5LargeV2Adapter(_SentenceTransformerAdapter):
    """E5: formatos asimétricos ``query:`` y ``passage:`` sin duplicarlos."""

    @staticmethod
    def _prefix_once(text: str, prefix: str) -> str:
        return text if text.startswith(prefix) else f"{prefix}{text}"

    def format_query(self, text: str) -> str:
        return self._prefix_once(text, self.config.query_prefix)

    def format_documents(self, texts: Sequence[str]) -> list[str]:
        return [
            self._prefix_once(text, self.config.document_prefix)
            for text in texts
        ]


class Qwen3EmbeddingAdapter(_SentenceTransformerAdapter):
    """Qwen3 0.6B: instrucción solo en queries y documentos originales."""

    def format_query(self, text: str) -> str:
        return text if text.startswith(self.config.query_prefix) else (
            f"{self.config.query_prefix}{text}"
        )

    def format_documents(self, texts: Sequence[str]) -> list[str]:
        return list(texts)

    def _model_source(self) -> str:
        """Resuelve el snapshot parcial controlado sin depender de red."""
        from pathlib import Path

        from huggingface_hub import try_to_load_from_cache

        config_path = try_to_load_from_cache(
            self.config.model_name,
            "config.json",
            revision=self.config.model_revision,
        )
        if not isinstance(config_path, str):
            raise RuntimeError(
                f"Snapshot local incompleto para {self.config.model_name!r} "
                f"en la revisión {self.config.model_revision}"
            )
        return str(Path(config_path).parent)

    def _postprocess_matrix(self, matrix: FloatMatrix) -> FloatMatrix:
        # Qwen se carga en BF16 en CPU; se reafirma L2 tras convertir a float32.
        if not self.config.normalize_embeddings:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Qwen produjo un embedding con norma cero")
        return np.asarray(matrix / norms, dtype=np.float32)

    def metadata(self) -> dict[str, Any]:
        result = super().metadata()
        encoder = self._encoder
        if encoder is None:
            return result
        model_config = encoder[0].auto_model.config
        pooling = encoder[1]
        pooling_config = (
            pooling.get_config_dict()
            if hasattr(pooling, "get_config_dict")
            else None
        )
        result.update({
            "parameter_count": sum(
                parameter.numel() for parameter in encoder.parameters()
            ),
            "declared_context_length": getattr(
                model_config, "max_position_embeddings", None
            ),
            "tokenizer_model_max_length": getattr(
                encoder.tokenizer, "model_max_length", None
            ),
            "pooling": pooling_config,
            "mrl_supported": True,
            "native_dimension": 1024,
        })
        return result


def adapter_for_config(config: ExperimentConfig) -> EmbeddingAdapter:
    """Selecciona explícitamente entre las tres familias implementadas."""
    if config.model_name.startswith("BAAI/bge-"):
        return BgeV15Adapter(config)
    if config.model_name == "intfloat/e5-large-v2":
        return E5LargeV2Adapter(config)
    if config.model_name == "Qwen/Qwen3-Embedding-0.6B":
        return Qwen3EmbeddingAdapter(config)
    raise ValueError(f"Modelo sin adaptador experimental: {config.model_name}")
