# Benchmark de retrieval v2 (Dani) — FROZEN

Copia local del question set de
`dani/experiments/benchmark/retrieval_benchmark_v2.jsonl`
(rama `experiment/dani-embeddings-chunking`).

- 48 preguntas, queries en español, corpus SEC en inglés
- Ground truth = EvidenceSpan (`char_start`/`char_end`), no `chunk_id`
- SHA-256: `6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C`

Evaluación con el retriever de Cristian (`miax_s1.buscar`):

```bash
python -m cristian.experiments.eval_retrieval_benchmark_v2
```
