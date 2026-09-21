"""Tests offline de las ablaciones controladas de embeddings."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
import unittest

import numpy as np

from dani.experiments.config import (
    ExperimentConfig,
    QWEN3_TASK_DESCRIPTION,
    e1_bge_large_config,
    e2_e5_large_v2_config,
    e3_qwen3_embedding_06b_config,
)
from dani.experiments.embeddings import (
    BgeV15Adapter,
    E5LargeV2Adapter,
    Qwen3EmbeddingAdapter,
)
from dani.experiments.runner import default_output_path


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: Sequence[str], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(list(texts))
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
        self.assertEqual(
            config.model_revision,
            "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        )
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
            {
                "experiment_id", "model_name", "model_revision",
                "expected_dimension",
            },
        )
        self.assertNotEqual(default_output_path(ExperimentConfig()),
                            default_output_path(e1_bge_large_config()))
        self.assertEqual(ExperimentConfig().chunk_source,
                         e1_bge_large_config().chunk_source)
        self.assertEqual(ExperimentConfig().evidence_path,
                         e1_bge_large_config().evidence_path)

    def test_e2_configuration_and_canonical_formatting(self) -> None:
        config = e2_e5_large_v2_config()
        encoder = FakeEncoder(1024)
        adapter = E5LargeV2Adapter(config, encoder=encoder)
        adapter.load()
        query = adapter.encode_query("test")
        documents = adapter.encode_documents(["test"])

        self.assertEqual(config.model_name, "intfloat/e5-large-v2")
        self.assertEqual(
            config.model_revision,
            "f169b11e22de13617baa190a028a32f3493550b6",
        )
        self.assertEqual(config.expected_dimension, 1024)
        self.assertEqual(config.query_prefix, "query: ")
        self.assertEqual(config.document_prefix, "passage: ")
        self.assertTrue(config.normalize_embeddings)
        self.assertEqual(query.shape, (1, 1024))
        self.assertEqual(documents.shape, (1, 1024))
        self.assertEqual(encoder.calls, [["query: test"], ["passage: test"]])

    def test_e5_prefixes_are_not_duplicated(self) -> None:
        encoder = FakeEncoder(1024)
        adapter = E5LargeV2Adapter(e2_e5_large_v2_config(), encoder=encoder)
        adapter.encode_query("query: test")
        adapter.encode_documents(["passage: test", "raw"])
        self.assertEqual(
            encoder.calls,
            [["query: test"], ["passage: test", "passage: raw"]],
        )

    def test_e3_configuration_and_exact_query_formatting(self) -> None:
        config = e3_qwen3_embedding_06b_config()
        encoder = FakeEncoder(1024, max_seq_length=32768)
        adapter = Qwen3EmbeddingAdapter(config, encoder=encoder)
        adapter.load()
        adapter.encode_query("abc")
        adapter.encode_documents(["abc"])

        expected_query = (
            "Instruct: Given a financial question, retrieve relevant passages "
            "from SEC 10-K filings that answer the question.\nQuery:abc"
        )
        self.assertEqual(config.model_name, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(config.expected_dimension, 1024)
        self.assertEqual(config.expected_max_sequence_length, 32768)
        self.assertEqual(config.requested_device, "cuda")
        self.assertTrue(config.normalize_embeddings)
        self.assertEqual(config.query_prefix, f"Instruct: {QWEN3_TASK_DESCRIPTION}\nQuery:")
        self.assertEqual(encoder.calls, [[expected_query], ["abc"]])

    def test_qwen_instruction_is_not_duplicated_and_documents_are_raw(self) -> None:
        config = e3_qwen3_embedding_06b_config()
        encoder = FakeEncoder(1024, max_seq_length=32768)
        adapter = Qwen3EmbeddingAdapter(config, encoder=encoder)
        formatted = f"{config.query_prefix}abc"
        adapter.encode_query(formatted)
        adapter.encode_documents(["abc", formatted])
        self.assertEqual(encoder.calls[0], [formatted])
        self.assertEqual(encoder.calls[1], ["abc", formatted])
        self.assertEqual(encoder.calls[0][0].count("Instruct:"), 1)
        self.assertEqual(encoder.calls[0][0].count("Query:"), 1)

    def test_bge_formatting_regression(self) -> None:
        for config in (ExperimentConfig(), e1_bge_large_config()):
            adapter = BgeV15Adapter(
                config, encoder=FakeEncoder(config.expected_dimension)
            )
            self.assertEqual(
                adapter.format_query("test"), f"{config.query_prefix}test"
            )
            self.assertEqual(adapter.format_documents(["test"]), ["test"])

    def test_e0_e1_e2_e3_share_non_embedding_configuration(self) -> None:
        configurations = [
            asdict(ExperimentConfig()),
            asdict(e1_bge_large_config()),
            asdict(e2_e5_large_v2_config()),
            asdict(e3_qwen3_embedding_06b_config()),
        ]
        allowed = {
            "experiment_id",
            "model_name",
            "model_revision",
            "query_prefix",
            "document_prefix",
            "document_format",
            "expected_dimension",
            "expected_max_sequence_length",
            "requested_device",
        }
        for candidate in configurations[1:]:
            differences = {
                key
                for key in configurations[0]
                if configurations[0][key] != candidate[key]
            }
            self.assertLessEqual(differences, allowed)
        self.assertEqual(
            {
                key
                for key in configurations[0]
                if configurations[0][key] != configurations[2][key]
            },
            allowed - {
                "expected_max_sequence_length",
                "requested_device",
            },
        )

    def test_token_lengths_are_measured_before_truncation(self) -> None:
        adapter = BgeV15Adapter(
            e1_bge_large_config(), encoder=FakeEncoder(1024)
        )
        self.assertEqual(adapter.token_lengths(["one two", "three"]), [4, 3])

    def test_e5_token_lengths_use_passage_format(self) -> None:
        encoder = FakeEncoder(1024)
        adapter = E5LargeV2Adapter(e2_e5_large_v2_config(), encoder=encoder)
        adapter.token_lengths(["one two"])
        self.assertEqual(encoder.tokenizer.calls, [["passage: one two"]])

    def test_effective_model_metadata_mismatch_fails(self) -> None:
        adapter = BgeV15Adapter(
            e1_bge_large_config(), encoder=FakeEncoder(384)
        )
        with self.assertRaisesRegex(ValueError, "Dimensión efectiva"):
            adapter.load()


if __name__ == "__main__":
    unittest.main()
