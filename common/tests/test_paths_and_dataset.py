from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from common.config import (DatasetConfigurationError, clear_dataset_path_cache,
                           get_dataset_paths)


EXPECTED_HASHES = {
    "chunks.jsonl": "388ff3671742c2248e8f5cb1c75afbc786edfa0dc6f72a62d1fadf2310ec82b2",
    "secciones.jsonl": "823272082bcf84fb14f6de415d087f2ee708b77dc475bcf3c3ff1a2d1e487dd1",
    "xbrl_facts.parquet": "f802fc89c2dba96dfef3dd2ef5025e5540620358fc2944e5303093a929127610",
    "chunks_meta.parquet": "fbd22360e517e5da250b18f0c95dbc9ad704b41902cedb1154747fde0be16fd9",
    "corpus.faiss": "605ce725de4604a3cd37a35ee61a03d52aa9351543d8ca98f53d5535dc9e8f5a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PathTests(unittest.TestCase):
    def tearDown(self):
        clear_dataset_path_cache()

    def test_real_dataset_is_auto_detected(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MIAX_DATASET_DIR", None)
            clear_dataset_path_cache()
            paths = get_dataset_paths()
        self.assertEqual(paths.dataset_dir.name, "dataset")
        self.assertEqual(paths.corpus_dir.name, "corpus_miax_2026")
        self.assertEqual(paths.index_dir.name, "indice_faiss")

    def test_env_is_authoritative_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MIAX_DATASET_DIR": directory}):
                clear_dataset_path_cache()
                with self.assertRaises(DatasetConfigurationError):
                    get_dataset_paths()


class DatasetIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clear_dataset_path_cache()
        cls.paths = get_dataset_paths()

    def test_frozen_hashes(self):
        actual = {name: sha256(path) for name, path
                  in self.paths.required_files().items()}
        self.assertEqual(actual, EXPECTED_HASHES)

    def test_counts_and_alignment(self):
        with self.paths.sections.open(encoding="utf-8") as stream:
            sections = [json.loads(line) for line in stream if line.strip()]
        with self.paths.chunks.open(encoding="utf-8") as stream:
            chunks = [json.loads(line) for line in stream if line.strip()]
        metadata = pd.read_parquet(self.paths.chunks_meta)
        facts = pd.read_parquet(self.paths.xbrl_facts)
        self.assertEqual((len(sections), len(chunks), len(metadata), len(facts)),
                         (48, 1749, 1749, 135))
        self.assertEqual([row["chunk_id"] for row in chunks],
                         metadata["chunk_id"].tolist())

    def test_faiss_header_without_importing_faiss(self):
        with self.paths.faiss_index.open("rb") as stream:
            header = stream.read(16)
        self.assertEqual(header[:4], b"IxFI")
        self.assertEqual(struct.unpack("<i", header[4:8])[0], 384)
        self.assertEqual(struct.unpack("<q", header[8:16])[0], 1749)
