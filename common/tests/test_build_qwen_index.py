from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from common.retrieval.dense_baseline import CHUNKS_META_SHA256, CHUNKS_SHA256
from common.retrieval.profiles import get_embedding_profile
from common.scripts import build_qwen_index as builder


class QwenIndexBuilderTests(unittest.TestCase):
    def test_manifest_records_generated_sha_and_frozen_contract(self):
        manifest = builder._manifest(
            get_embedding_profile("qwen3-06b"), "a" * 64
        )
        self.assertEqual(manifest["faiss_sha256"], "a" * 64)
        self.assertEqual(manifest["chunks_sha256"], CHUNKS_SHA256)
        self.assertEqual(manifest["chunks_meta_sha256"], CHUNKS_META_SHA256)
        self.assertEqual(manifest["pooling"], "lasttoken")
        self.assertTrue(manifest["include_prompt"])
        self.assertTrue(manifest["normalize"])
        self.assertEqual(manifest["document_format"], "raw")
        self.assertEqual(manifest["index_type"], "IndexFlatIP")
        self.assertEqual(manifest["ntotal"], 1749)

    def test_float32_l2_is_reasserted(self):
        vectors = np.zeros((2, 1024), dtype=np.float32)
        vectors[0, 0] = 0.998
        vectors[1, 1] = 2.0
        with patch.object(builder, "EXPECTED_CHUNKS", 2):
            matrix = builder._float32_l2(vectors)
        self.assertEqual(matrix.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0)

    def test_chunks_and_metadata_must_have_same_count_order_and_text(self):
        records = [
            {"chunk_id": "c0", "texto": "zero"},
            {"chunk_id": "c1", "texto": "one"},
        ]
        metadata = pd.DataFrame(records)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with patch.object(builder, "EXPECTED_CHUNKS", 2):
                self.assertEqual(
                    builder._read_documents(path, metadata), ["zero", "one"]
                )
                with self.assertRaisesRegex(RuntimeError, "orden"):
                    builder._read_documents(path, metadata.iloc[::-1])


if __name__ == "__main__":
    unittest.main()
