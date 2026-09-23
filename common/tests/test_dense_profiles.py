from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from common.config import DatasetPaths, _build_paths
from common.retrieval import dense_baseline as dense
from common.retrieval.profiles import RetrievalProfileError, get_embedding_profile

VALIDATED_QWEN_FAISS_SHA256 = (
    "f2180a18b804b1e767e34d1ad84bed3cb3af07c9206e488f3fd6d3cfd872d2de"
)


class FakePooling:
    def __init__(self, mode="lasttoken", include_prompt=True):
        self.config = {
            "embedding_dimension": 1024,
            "pooling_mode": mode,
            "include_prompt": include_prompt,
        }

    def get_config_dict(self):
        return self.config


class FakeEncoder:
    def __init__(self, dimension, *, normalized=True, pooling=None):
        self.dimension = dimension
        self.normalized = normalized
        self.pooling = pooling or FakePooling()
        self.calls = []

    def __getitem__(self, position):
        if position == 1:
            return self.pooling
        raise IndexError(position)

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        if isinstance(self.normalized, bool):
            scale = 1.0 if self.normalized else 2.0
        else:
            scale = float(self.normalized)
        vectors[:, 0] = scale
        return vectors


class IndexFlatIP:
    def __init__(self, dimension, scores=(0.75,), positions=(0,), ntotal=None):
        self.d = dimension
        self.scores = np.asarray([scores], dtype=np.float32)
        self.positions = np.asarray([positions], dtype=np.int64)
        self.ntotal = len(positions) if ntotal is None else ntotal

    def search(self, vector, k):
        self.last_vector = vector.copy()
        return self.scores, self.positions


def qwen_manifest():
    profile = get_embedding_profile("qwen3-06b")
    return {
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
        "faiss_sha256": VALIDATED_QWEN_FAISS_SHA256,
        "chunks_sha256": dense.CHUNKS_SHA256,
        "chunks_meta_sha256": dense.CHUNKS_META_SHA256,
    }


class ProfileTests(unittest.TestCase):
    def test_profiles_are_exact_and_use_separate_indexes(self):
        bge = get_embedding_profile("bge-small-baseline")
        qwen = get_embedding_profile("qwen3-06b")
        self.assertEqual(bge.model_name, "BAAI/bge-small-en-v1.5")
        self.assertIsNone(bge.model_revision)
        self.assertEqual(bge.dimension, 384)
        self.assertEqual(qwen.model_name, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(
            qwen.model_revision,
            "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        )
        self.assertEqual(qwen.dimension, 1024)
        self.assertNotEqual(bge.index_dirname, qwen.index_dirname)

    def test_bge_always_adds_prefix_and_qwen_format_is_exact(self):
        bge = get_embedding_profile("bge-small-baseline")
        prefix = bge.query_prefix
        self.assertEqual(bge.format_query("raw"), prefix + "raw")
        self.assertEqual(bge.format_query(prefix + "raw"), prefix + prefix + "raw")
        qwen = get_embedding_profile("qwen3-06b")
        self.assertEqual(qwen.format_query("pregunta"), qwen.query_prefix + "pregunta")

    def test_unknown_profile_fails_closed(self):
        with self.assertRaises(RetrievalProfileError):
            get_embedding_profile("unknown")

    def test_qwen_encoder_is_pinned_and_bge_call_is_unchanged(self):
        factory = Mock(return_value=object())
        dense._create_encoder(get_embedding_profile("bge-small-baseline"), factory)
        factory.assert_called_with("BAAI/bge-small-en-v1.5")
        factory.reset_mock()
        dense._create_encoder(get_embedding_profile("qwen3-06b"), factory)
        factory.assert_called_with(
            "Qwen/Qwen3-Embedding-0.6B",
            revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            local_files_only=True,
        )

    def test_paths_require_manifest_only_for_qwen(self):
        with tempfile.TemporaryDirectory() as directory:
            bge = _build_paths(Path(directory), "bge-small-baseline")
            qwen = _build_paths(Path(directory), "qwen3-06b")
        self.assertIsNone(bge.index_manifest)
        self.assertEqual(qwen.index_manifest.name, "index_manifest.json")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        corpus = root / "corpus_miax_2026"
        index_dir = root / "indice_faiss_qwen3_06b"
        corpus.mkdir()
        index_dir.mkdir()
        self.paths = DatasetPaths(
            dataset_dir=root,
            corpus_dir=corpus,
            index_dir=index_dir,
            sections=corpus / "secciones.jsonl",
            chunks=corpus / "chunks.jsonl",
            xbrl_facts=corpus / "xbrl_facts.parquet",
            faiss_index=index_dir / "corpus.faiss",
            chunks_meta=index_dir / "chunks_meta.parquet",
            index_manifest=index_dir / "index_manifest.json",
        )
        for path in (self.paths.chunks, self.paths.faiss_index, self.paths.chunks_meta):
            path.write_bytes(b"fixture")
        self.write_manifest(qwen_manifest())

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self, value):
        self.paths.index_manifest.write_text(json.dumps(value), encoding="utf-8")

    def valid_hash(self, path):
        return {
            self.paths.faiss_index: self.current_manifest()["faiss_sha256"],
            self.paths.chunks: dense.CHUNKS_SHA256,
            self.paths.chunks_meta: dense.CHUNKS_META_SHA256,
        }[path]

    def validate(self):
        profile = get_embedding_profile("qwen3-06b")
        with patch.object(dense, "_sha256", side_effect=self.valid_hash):
            return dense._validate_qwen_manifest(profile, self.paths)

    def current_manifest(self):
        return json.loads(self.paths.index_manifest.read_text(encoding="utf-8"))

    def test_valid_flat_manifest_is_accepted(self):
        self.assertEqual(self.validate(), qwen_manifest())

    def test_rebuilt_index_sha_is_accepted_when_file_matches_manifest(self):
        manifest = qwen_manifest()
        manifest["faiss_sha256"] = "a" * 64
        self.write_manifest(manifest)
        self.assertEqual(self.validate(), manifest)

    def test_absent_manifest_fails_closed(self):
        self.paths.index_manifest.unlink()
        with self.assertRaises(dense.DenseRetrievalConfigurationError):
            self.validate()

    def test_required_manifest_fields_fail_closed(self):
        invalid = {
            "schema_version": 2,
            "model_revision": "wrong",
            "query_template": "wrong",
        }
        for field, value in invalid.items():
            manifest = qwen_manifest()
            manifest[field] = value
            self.write_manifest(manifest)
            with self.subTest(field=field):
                with self.assertRaises(dense.DenseRetrievalConfigurationError):
                    self.validate()

    def test_each_artifact_hash_mismatch_fails_closed(self):
        for path in (self.paths.faiss_index, self.paths.chunks, self.paths.chunks_meta):
            def wrong_hash(candidate, wrong=path):
                return "0" * 64 if candidate == wrong else self.valid_hash(candidate)

            with self.subTest(path=path.name), patch.object(
                dense, "_sha256", side_effect=wrong_hash
            ):
                with self.assertRaises(dense.DenseRetrievalConfigurationError):
                    dense._validate_qwen_manifest(
                        get_embedding_profile("qwen3-06b"), self.paths
                    )

    def test_invalid_manifest_is_checked_before_encoder_creation(self):
        with patch.object(
            dense, "get_dataset_paths", return_value=self.paths
        ), patch.object(
            dense, "_validate_qwen_manifest", side_effect=RuntimeError("invalid")
        ), patch.object(dense, "_create_encoder") as create:
            dense.clear_dense_retrieval_cache()
            with self.assertRaisesRegex(RuntimeError, "invalid"):
                dense._load_profile_resources("qwen3-06b")
            create.assert_not_called()


class ResourceValidationTests(unittest.TestCase):
    def test_realistic_qwen_pooling_is_accepted(self):
        dense._validate_encoder(
            get_embedding_profile("qwen3-06b"), FakeEncoder(1024)
        )

    def test_wrong_qwen_pooling_or_prompt_fails(self):
        for pooling in (
            FakePooling(mode="mean"),
            FakePooling(include_prompt=False),
        ):
            with self.subTest(config=pooling.config):
                with self.assertRaises(dense.DenseRetrievalConfigurationError):
                    dense._validate_encoder(
                        get_embedding_profile("qwen3-06b"),
                        FakeEncoder(1024, pooling=pooling),
                    )

    def test_sentence_transformers_lasttoken_uses_attention_mask(self):
        import torch
        from sentence_transformers.sentence_transformer.modules import Pooling

        pooling = Pooling(2, pooling_mode="lasttoken", include_prompt=True)
        tokens = torch.tensor([
            [[1., 1.], [2., 2.], [9., 9.], [9., 9.]],
            [[9., 9.], [9., 9.], [3., 3.], [4., 4.]],
        ])
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
        output = pooling({
            "token_embeddings": tokens,
            "attention_mask": mask,
        })["sentence_embedding"]
        self.assertTrue(torch.equal(output, torch.tensor([[2., 2.], [4., 4.]])))

    def test_qwen_rejects_bge_dimension(self):
        with self.assertRaises(dense.DenseRetrievalConfigurationError):
            dense._validate_index(
                get_embedding_profile("qwen3-06b"),
                IndexFlatIP(384),
                [object()],
                {"ntotal": 1},
            )

    def test_bge_does_not_need_qwen_manifest(self):
        profile = get_embedding_profile("bge-small-baseline")
        dense._validate_index(
            profile, IndexFlatIP(384), [object()], manifest=None
        )
        dense._validate_encoder(profile, FakeEncoder(384))

    def test_non_normalized_query_fails(self):
        profile = get_embedding_profile("qwen3-06b")
        resources = (
            IndexFlatIP(1024),
            metadata_fixture().iloc[:1],
            FakeEncoder(1024, normalized=False),
            profile,
        )
        with patch.object(dense, "_indice", return_value=resources):
            with self.assertRaises(dense.DenseRetrievalConfigurationError):
                dense.buscar("query")

    def test_qwen_reasserts_l2_after_float32_cast(self):
        profile = get_embedding_profile("qwen3-06b")
        encoder = FakeEncoder(1024)
        encoder.normalized = 0.998
        index = IndexFlatIP(1024)
        resources = (index, metadata_fixture().iloc[:1], encoder, profile)
        with patch.object(dense, "_indice", return_value=resources):
            dense.buscar("query")
        self.assertAlmostEqual(float(np.linalg.norm(index.last_vector[0])), 1.0)


def metadata_fixture():
    return pd.DataFrame([
        {"chunk_id": "c0", "ticker": "AAPL", "fiscal_year": 2025,
         "item": "7", "texto": "zero", "n_tokens": 1,
         "contiene_tabla": False},
        {"chunk_id": "c1", "ticker": "AAPL", "fiscal_year": 2025,
         "item": "8", "texto": "one", "n_tokens": 1,
         "contiene_tabla": False},
        {"chunk_id": "c2", "ticker": "MSFT", "fiscal_year": 2025,
         "item": "7", "texto": "two", "n_tokens": 1,
         "contiene_tabla": False},
        {"chunk_id": "c3", "ticker": "AAPL", "fiscal_year": 2024,
         "item": "7", "texto": "three", "n_tokens": 1,
         "contiene_tabla": True},
    ])


def baseline_buscar(index, metadata, encoder, query, ticker=None,
                    fiscal_year=None, item=None, k=5):
    prefix = "Represent this sentence for searching relevant passages: "
    vector = encoder.encode(
        [prefix + query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
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
            "chunk_id": row["chunk_id"], "ticker": row["ticker"],
            "fiscal_year": int(row["fiscal_year"]), "item": row["item"],
            "texto": row["texto"], "n_tokens": int(row["n_tokens"]),
            "contiene_tabla": bool(row["contiene_tabla"]),
            "puntuacion": round(float(score), 4),
        })
        if len(results) >= k:
            break
    return results


class BgeDifferentialTests(unittest.TestCase):
    def test_baseline_and_profile_are_exact_for_queries_ties_and_filters(self):
        profile = get_embedding_profile("bge-small-baseline")
        metadata = metadata_fixture()
        cases = (
            ("raw", None, None, None, 4),
            (profile.query_prefix + "already", None, None, None, 4),
            ("filtered", "AAPL", 2025, "7", 5),
            ("year", "AAPL", 2024, None, 5),
        )
        for query, ticker, year, item, k in cases:
            expected_encoder = FakeEncoder(384)
            actual_encoder = FakeEncoder(384)
            expected_index = IndexFlatIP(
                384, scores=(0.9, 0.9, 0.7, 0.6), positions=(0, 1, 2, 3)
            )
            actual_index = IndexFlatIP(
                384, scores=(0.9, 0.9, 0.7, 0.6), positions=(0, 1, 2, 3)
            )
            expected = baseline_buscar(
                expected_index, metadata, expected_encoder, query,
                ticker, year, item, k,
            )
            resources = (actual_index, metadata, actual_encoder, profile)
            with self.subTest(query=query, item=item), patch.object(
                dense, "_indice", return_value=resources
            ):
                actual = dense.buscar(query, ticker, year, item, k)
                self.assertEqual(actual, expected)
                self.assertEqual(
                    json.dumps(actual, ensure_ascii=False),
                    json.dumps(expected, ensure_ascii=False),
                )
                self.assertEqual(
                    dense.formatear_fragmentos(actual),
                    dense.formatear_fragmentos(expected),
                )
                self.assertEqual(actual_encoder.calls, expected_encoder.calls)


if __name__ == "__main__":
    unittest.main()
