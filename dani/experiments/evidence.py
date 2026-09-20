"""Ground truth de evidencia expresado sobre las secciones fuente."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from common.config import get_dataset_paths
from dani.experiments.config import ExperimentConfig

EXPERIMENT_ROOT = Path(__file__).resolve().parent


class EvidenceResolutionError(ValueError):
    """Un ancla no puede resolverse de forma inequívoca."""

    def __init__(self, status: str, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status


@dataclass(frozen=True)
class EvidenceSpan:
    """Evidencia en coordenadas de una sección fuente.

    Los offsets siguen la semántica de slices de Python: ``char_start`` es
    inclusivo y ``char_end`` exclusivo.
    """

    question_id: str
    ticker: str
    fiscal_year: int
    item: str
    evidence_text: str
    char_start: int
    char_end: int
    matching_method: str
    source_text_span: str | None = None

    def __post_init__(self) -> None:
        if self.char_start < 0:
            raise ValueError("char_start no puede ser negativo")
        if self.char_end <= self.char_start:
            raise ValueError("char_end debe ser mayor que char_start")
        if not self.evidence_text:
            raise ValueError("evidence_text no puede estar vacío")
        expected_span = self.source_text_span or self.evidence_text
        if len(expected_span) != self.char_end - self.char_start:
            raise ValueError("La longitud del span no coincide con sus offsets")

    def validate_against(self, source_text: str) -> None:
        expected = self.source_text_span or self.evidence_text
        actual = source_text[self.char_start:self.char_end]
        if actual != expected:
            raise ValueError(
                f"{self.question_id}: source[start:end] no coincide con el span"
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.source_text_span is None:
            result.pop("source_text_span")
        return result


class EvidenceMapper:
    """Resuelve anclas en secciones y proyecta spans sobre chunks."""

    def __init__(
        self,
        sections: Iterable[dict[str, Any]],
        chunks: Iterable[dict[str, Any]],
    ):
        self.sections = list(sections)
        self.chunks = list(chunks)
        self._sections_by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for section in self.sections:
            key = self._key(section)
            self._sections_by_key.setdefault(key, []).append(section)

    @classmethod
    def from_dataset(cls) -> "EvidenceMapper":
        paths = get_dataset_paths()
        return cls(_read_jsonl(paths.sections), _read_jsonl(paths.chunks))

    def resolve(self, question: dict[str, Any]) -> EvidenceSpan:
        section = self.section_for(question)
        source_text = str(section["texto"])
        anchor = str(question["ancla_texto"])
        exact_positions = _find_occurrences(source_text, anchor)
        if len(exact_positions) > 1:
            raise EvidenceResolutionError(
                "MULTIPLE",
                f"{question['id']}: {len(exact_positions)} matches literales",
            )
        if len(exact_positions) == 1:
            start = exact_positions[0]
            span = EvidenceSpan(
                question_id=str(question["id"]),
                ticker=str(question["ticker"]),
                fiscal_year=int(question["fiscal_year"]),
                item=self._question_item(question),
                evidence_text=anchor,
                char_start=start,
                char_end=start + len(anchor),
                matching_method="exact_literal",
            )
            span.validate_against(source_text)
            return span

        normalized_source, offset_map = _normalize_whitespace_with_map(source_text)
        normalized_anchor = _normalize_whitespace(anchor)
        normalized_positions = _find_occurrences(
            normalized_source, normalized_anchor
        )
        if len(normalized_positions) > 1:
            raise EvidenceResolutionError(
                "MULTIPLE",
                f"{question['id']}: {len(normalized_positions)} matches "
                "tras normalizar whitespace",
            )
        if not normalized_positions:
            raise EvidenceResolutionError(
                "NOT_FOUND", f"{question['id']}: ancla ausente en la sección"
            )
        normalized_start = normalized_positions[0]
        normalized_end = normalized_start + len(normalized_anchor)
        start = offset_map[normalized_start][0]
        end = offset_map[normalized_end - 1][1]
        source_span = source_text[start:end]
        span = EvidenceSpan(
            question_id=str(question["id"]),
            ticker=str(question["ticker"]),
            fiscal_year=int(question["fiscal_year"]),
            item=self._question_item(question),
            evidence_text=anchor,
            char_start=start,
            char_end=end,
            matching_method="whitespace_normalized",
            source_text_span=source_span,
        )
        span.validate_against(source_text)
        return span

    def project(self, span: EvidenceSpan) -> dict[str, Any]:
        section = self.section_for(span)
        source_text = str(section["texto"])
        span.validate_against(source_text)
        full: list[str] = []
        partial: list[str] = []
        for chunk in self.chunks:
            if self._key(chunk) != self._key(span):
                continue
            start = int(chunk["inicio_car"])
            end = int(chunk["fin_car"])
            self._validate_chunk_slice(chunk, source_text, start, end)
            contains = start <= span.char_start and end >= span.char_end
            overlaps = start < span.char_end and end > span.char_start
            if contains:
                full.append(str(chunk["chunk_id"]))
            elif overlaps:
                partial.append(str(chunk["chunk_id"]))
        status = (
            "full_containment" if full
            else "partial_overlap" if partial
            else "no_overlap"
        )
        return {
            "status": status,
            "full_containment_chunk_ids": full,
            "partial_overlap_chunk_ids": partial,
        }

    def section_for(self, record: Any) -> dict[str, Any]:
        matches = self._sections_by_key.get(self._key(record), [])
        if len(matches) != 1:
            raise EvidenceResolutionError(
                "SOURCE_SECTION_ERROR",
                f"esperada una sección para {self._key(record)}, hay {len(matches)}",
            )
        return matches[0]

    def validate_chunk_offsets(self) -> int:
        for chunk in self.chunks:
            source_text = str(self.section_for(chunk)["texto"])
            self._validate_chunk_slice(
                chunk,
                source_text,
                int(chunk["inicio_car"]),
                int(chunk["fin_car"]),
            )
        return len(self.chunks)

    @staticmethod
    def _question_item(question: dict[str, Any]) -> str:
        return str(question.get("item", question.get("item_esperado")))

    @classmethod
    def _key(cls, record: Any) -> tuple[str, int, str]:
        if isinstance(record, EvidenceSpan):
            return record.ticker, record.fiscal_year, record.item
        item = record.get("item", record.get("item_esperado"))
        return str(record["ticker"]), int(record["fiscal_year"]), str(item)

    @staticmethod
    def _validate_chunk_slice(
        chunk: dict[str, Any], source_text: str, start: int, end: int
    ) -> None:
        if start < 0 or end <= start:
            raise ValueError(f"Offsets inválidos en {chunk['chunk_id']}")
        if source_text[start:end] != str(chunk["texto"]):
            raise ValueError(
                f"{chunk['chunk_id']}: inicio_car/fin_car no describen el texto"
            )


def build_ground_truth_artifact() -> dict[str, Any]:
    """Genera la fuente de verdad y audita su paridad con E0."""
    import pandas as pd

    from dani.experiments.evaluation import RetrievalEvaluator

    paths = get_dataset_paths()
    config = ExperimentConfig()
    sections = _read_jsonl(paths.sections)
    chunks = _read_jsonl(paths.chunks)
    questions = [
        question for question in _read_jsonl(config.golden_path)
        if question.get("ancla_texto")
    ]
    mapper = EvidenceMapper(sections, chunks)
    mapper.validate_chunk_offsets()
    legacy = RetrievalEvaluator(
        config.golden_path, pd.DataFrame(chunks), max_k=config.max_k
    )

    evidence: list[dict[str, Any]] = []
    exact_unique = normalized = split = parity_count = 0
    for question in questions:
        span = mapper.resolve(question)
        projection = mapper.project(span)
        old_relevant = legacy.relevant_chunk_ids(question)
        span_relevant = projection["full_containment_chunk_ids"]
        same_set = set(old_relevant) == set(span_relevant)
        exact_unique += span.matching_method == "exact_literal"
        normalized += span.matching_method == "whitespace_normalized"
        split += not span_relevant and bool(
            projection["partial_overlap_chunk_ids"]
        )
        parity_count += same_set
        evidence.append({
            **span.to_dict(),
            "original_relevant_chunk_ids": span_relevant,
            "e0_relevant_chunk_ids": old_relevant,
            "partial_overlap_chunk_ids": projection[
                "partial_overlap_chunk_ids"
            ],
            "same_e0_relevant_set": same_set,
        })

    return {
        "schema_version": "1",
        "source": {
            "golden_sha256": _sha256_file(config.golden_path),
            "sections_sha256": _sha256_file(paths.sections),
            "chunks_sha256": _sha256_file(paths.chunks),
        },
        "policy": {
            "offset_semantics": (
                "Python character slice: char_start inclusive, char_end exclusive"
            ),
            "chunk_offset_semantics": (
                "section_text[inicio_car:fin_car] == chunk_text; fin_car exclusive"
            ),
            "matching_order": [
                "exact_literal",
                "whitespace_normalized_with_original_offset_map",
            ],
            "whitespace_normalization": (
                "collapse consecutive Unicode whitespace to one ASCII space "
                "and strip leading/trailing whitespace"
            ),
            "relevance_rule": "full_containment",
            "full_containment": (
                "chunk_start <= evidence_start and chunk_end >= evidence_end"
            ),
        },
        "summary": {
            "n_questions": len(questions),
            "exact_unique_matches": exact_unique,
            "normalized_matches": normalized,
            "ambiguous_matches": 0,
            "not_found": 0,
            "evidence_with_full_containment": sum(
                bool(row["original_relevant_chunk_ids"]) for row in evidence
            ),
            "split_evidence": split,
            "e0_relevant_set_parity": parity_count,
        },
        "evidence": evidence,
    }


def write_ground_truth(output_path: Path) -> dict[str, Any]:
    artifact = build_ground_truth_artifact()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return artifact


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _find_occurrences(text: str, needle: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        position = text.find(needle, start)
        if position < 0:
            return positions
        positions.append(position)
        start = position + 1


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_whitespace_with_map(
    text: str,
) -> tuple[str, list[tuple[int, int]]]:
    normalized: list[str] = []
    offsets: list[tuple[int, int]] = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            start = position
            while position < len(text) and text[position].isspace():
                position += 1
            if normalized and position < len(text):
                normalized.append(" ")
                offsets.append((start, position))
            continue
        normalized.append(text[position])
        offsets.append((position, position + 1))
        position += 1
    return "".join(normalized), offsets


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera ground truth de evidencia independiente del chunking"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            EXPERIMENT_ROOT / "results" / "evidence_ground_truth_v1.json"
        ),
    )
    args = parser.parse_args()
    artifact = write_ground_truth(args.output.resolve())
    print(json.dumps({
        "output": str(args.output.resolve()),
        "summary": artifact["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

