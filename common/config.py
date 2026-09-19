"""Resolución centralizada y perezosa de las rutas del dataset."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATASET_ENV_VAR = "MIAX_DATASET_DIR"
REPO_ROOT = Path(__file__).resolve().parents[1]


class DatasetConfigurationError(RuntimeError):
    """La ubicación del dataset no existe o está incompleta."""


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    corpus_dir: Path
    index_dir: Path
    sections: Path
    chunks: Path
    xbrl_facts: Path
    faiss_index: Path
    chunks_meta: Path

    def required_files(self) -> dict[str, Path]:
        return {
            "secciones.jsonl": self.sections,
            "chunks.jsonl": self.chunks,
            "xbrl_facts.parquet": self.xbrl_facts,
            "corpus.faiss": self.faiss_index,
            "chunks_meta.parquet": self.chunks_meta,
        }


def dataset_candidates() -> tuple[Path, ...]:
    """Candidatos permitidos, en orden y sin depender del CWD."""
    configured = os.getenv(DATASET_ENV_VAR)
    if configured:
        return (Path(configured).expanduser(),)
    return (REPO_ROOT / "dataset", REPO_ROOT.parent / "dataset")


def _build_paths(dataset_dir: Path) -> DatasetPaths:
    root = dataset_dir.resolve()
    corpus = root / "corpus_miax_2026"
    index = root / "indice_faiss"
    return DatasetPaths(
        dataset_dir=root,
        corpus_dir=corpus,
        index_dir=index,
        sections=corpus / "secciones.jsonl",
        chunks=corpus / "chunks.jsonl",
        xbrl_facts=corpus / "xbrl_facts.parquet",
        faiss_index=index / "corpus.faiss",
        chunks_meta=index / "chunks_meta.parquet",
    )


@lru_cache(maxsize=1)
def get_dataset_paths() -> DatasetPaths:
    """Resuelve y valida los cinco artefactos obligatorios del dataset.

    Si ``MIAX_DATASET_DIR`` está definida, esa ruta es autoritativa: un error
    no cae silenciosamente en otro dataset. Sin la variable se prueban
    ``repo/dataset`` y ``parent_del_repo/dataset``.
    """
    diagnostics: list[str] = []
    for candidate in dataset_candidates():
        paths = _build_paths(candidate)
        missing = [str(path) for path in paths.required_files().values()
                   if not path.is_file()]
        if not missing:
            return paths
        diagnostics.append(f"{paths.dataset_dir}: faltan {missing}")
    source = (f"{DATASET_ENV_VAR} definida" if os.getenv(DATASET_ENV_VAR)
              else "autodetección")
    raise DatasetConfigurationError(
        f"Dataset MIAX no encontrado o incompleto ({source}). "
        + " | ".join(diagnostics)
    )


def clear_dataset_path_cache() -> None:
    """Limpia la caché; útil para tests que cambian la variable de entorno."""
    get_dataset_paths.cache_clear()
