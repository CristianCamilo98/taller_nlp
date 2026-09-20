"""Tests offline de la ablación controlada BGE-small frente a BGE-large."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
import unittest

import numpy as np

from dani.experiments.config import ExperimentConfig, e1_bge_large_config
from dani.experiments.embeddings import BgeV15Adapter
from dani.experiments.runner import default_output_path


class FakeTokenizer:
    def __call__(self, texts: Sequence[str], **kwargs: Any) -> dict[str, Any]:
        return {"length": [len(text.split()) + 2 for text in texts]}


class FakeEncoder:
    def __init__(self, dimension: int, max_seq_length: int = 512):
        self.dimension = dimension
        self.max_seq_length = max_seq_length
        self.device = "cpu"
        self.tokenizer = FakeTokenizer()
        self.calls: list[list[str]] = []

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


class EmbeddingAblationConfigTests(unittest.TestCase):
    def test_e0_configuration_is_unchanged(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(config.experiment_id, "e0_bge_small_original")
        self.assertEqual(config.model_name, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.expected_dimension, 384)
        self.assertEqual(config.expected_max_sequence_length, 512)

    def test_e1_configuration_and_formatting(self) -> None:
        config = e1_bge_large_config()
        encoder = FakeEncoder(1024)
        adapter = BgeV15Adapter(config, encoder=encoder)
        adapter.load()
        query = adapter.encode_query("query")
        documents = adapter.encode_documents(["document"])

        self.assertEqual(config.model_name, "BAAI/bge-large-en-v1.5")
        self.assertEqual(config.expected_dimension, 1024)
        self.assertTrue(config.normalize_embeddings)
        self.assertEqual(query.shape, (1, 1024))
        self.assertEqual(documents.shape, (1, 1024))
        self.assertEqual(
            encoder.calls[0], [f"{config.query_prefix}query"]
        )
        self.assertEqual(encoder.calls[1], ["document"])

    def test_e0_and_e1_differ_only_in_embedding_identity(self) -> None:
        e0 = asdict(ExperimentConfig())
        e1 = asdict(e1_bge_large_config())
        differences = {key for key in e0 if e0[key] != e1[key]}
        self.assertEqual(
            differences,
            {"experiment_id", "model_name", "expected_dimension"},
        )
        self.assertNotEqual(default_output_path(ExperimentConfig()),
                            default_output_path(e1_bge_large_config()))
        self.assertEqual(ExperimentConfig().chunk_source,
                         e1_bge_large_config().chunk_source)
        self.assertEqual(ExperimentConfig().evidence_path,
                         e1_bge_large_config().evidence_path)

    def test_token_lengths_are_measured_before_truncation(self) -> None:
        adapter = BgeV15Adapter(
            e1_bge_large_config(), encoder=FakeEncoder(1024)
        )
        self.assertEqual(adapter.token_lengths(["one two", "three"]), [4, 3])

    def test_effective_model_metadata_mismatch_fails(self) -> None:
        adapter = BgeV15Adapter(
            e1_bge_large_config(), encoder=FakeEncoder(384)
        )
        with self.assertRaisesRegex(ValueError, "Dimensión efectiva"):
            adapter.load()


if __name__ == "__main__":
    unittest.main()
