# Benchmark de retrieval v2 — FROZEN

## Freeze

- **benchmark:** `retrieval_benchmark_v2.jsonl`
- **status:** `FROZEN`
- **número de preguntas:** 48
- **SHA-256:** `6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C`
- **queries:** español
- **corpus SEC:** inglés
- **table_adjacent:** exactamente 4
- **full containment con el chunking original:** 48/48

`retrieval_benchmark_v2.jsonl` es la única fuente de verdad del contenido del
benchmark. `benchmark_v2.py` contiene exclusivamente infraestructura genérica
de carga y validación; no replica preguntas, evidencias ni offsets.

La fuente documental canónica para comprobar cada EvidenceSpan es
`dataset/corpus_miax_2026/secciones.jsonl`. Los EvidenceSpan son independientes
del chunking y el benchmark no usa `chunk_id` como ground truth. Cada fila del
JSONL tiene este esquema:

- `question_id`, `ticker`, `fiscal_year`, `item`, `question`;
- `evidence_text`, `char_start`, `char_end`;
- `matching_method` (`exact_literal`);
- `evidence_type` (`prose`, `numeric` o `table_adjacent`).

Los offsets son intervalos semiabiertos `[char_start:char_end)` aplicados al
campo `texto` de la sección correspondiente y `matching_method` es siempre
`exact_literal`.

## Política de evaluación

La política se define antes de observar resultados sobre este benchmark:

- **FULL / ALL-48:** las 48 preguntas.
- **Por item:** 1A = 12, 7 = 12, 7A = 12 y 8 = 12.
- **DISCRIMINATIVE / NON-7A-36:** las 36 preguntas de los items 1A, 7 y 8.

Item 7A permanece en el benchmark completo porque es representativo del
sistema. Sin embargo, sus candidate pools son estructuralmente pequeños y en
algunos casos contienen un único chunk después del filtrado por ticker, año e
item, por lo que ofrecen poco poder discriminativo entre embeddings.

Las evaluaciones futuras reportarán siempre:

- métricas ALL-48;
- métricas por item;
- métricas NON-7A-36.

No se establecerán ganadores ni umbrales a partir de una inspección previa de
resultados de modelos.

## Política cross-lingual

Esta es una decisión experimental congelada, no un accidente:

- las queries del benchmark están en español;
- el corpus SEC está en inglés;
- **no query translation**: no se traduce la query;
- **no query rewriting**: no se reescribe la query;
- **no corpus translation**: no se traduce el corpus.

## Validación

La validación comprueba cobertura, schema, offsets exactos, unicidad de la
evidencia y ausencia de dependencia de chunks en el benchmark canónico:

```powershell
python -B -m dani.experiments.benchmark.benchmark_v2
```
