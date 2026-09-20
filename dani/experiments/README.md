# Experimento de retrieval de Dani — Bloque 1

## Pregunta de investigación

El trabajo estudiará embeddings y chunking mediante ablaciones controladas. Este
bloque construye solamente la infraestructura mínima y demuestra que reproduce
el control E0 antes de probar cualquier candidato.

## Control E0

E0 utiliza `BAAI/bge-small-en-v1.5`, prefijo BGE solo en la consulta,
documentos sin prefijo, normalización L2, los 1.749 chunks originales e
`IndexFlatIP`. Se obtiene primero el ranking denso global y después se filtra
por ticker, ejercicio e item. La evaluación usa las seis preguntas del golden
común que contienen `ancla_texto`.

## Arquitectura

```text
ExperimentConfig -> BgeV15Adapter -> DenseFaissRetriever
                                      |
RetrievalEvaluator -------------------+
                 -> ExperimentRunner -> resultado JSON
```

`RetrievalEvaluator` localiza evidencia mediante metadata y texto literal. El
`chunk_id` se reporta, pero no se usa por sí solo como verdad conceptual porque
dejará de ser estable cuando se experimente con chunking.

## Variables congeladas

- Modelo, formato de query/documentos y normalización.
- Chunks, orden, metadata e índice exacto `IndexFlatIP`.
- Ranking global seguido de postfiltrado.
- Recall@1/3/5/10 y MRR@10.

No forman parte de este bloque BM25, reranking, query rewriting, nuevos
chunkers, otros embeddings, cambios del agente ni llamadas a LLM.

## Reproducción offline

El entorno debe tener las dependencias comunes y el modelo completo en caché.
El adapter usa `local_files_only=True`; si falta el modelo, falla sin descargar.

```powershell
$env:MIAX_DATASET_DIR = "C:\ruta\al\dataset"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

python -B -m unittest discover -s dani/experiments/tests -v
python -B -m dani.experiments.runner
```

Se genera un índice regenerable en `artifacts/e0_bge_small_original/` y una
fuente de verdad legible en `results/e0_bge_small_original.json`, con manifest,
métricas y ranking por pregunta. El índice pesado está ignorado por Git.

## PASS/FAIL

El bloque pasa únicamente con seis preguntas y estas métricas:

| Métrica | Esperado |
| --- | ---: |
| Recall@1 | 0.16666666666666666 |
| Recall@3 | 0.3333333333333333 |
| Recall@5 | 0.6666666666666666 |
| Recall@10 | 0.8333333333333334 |
| MRR@10 | 0.3611111111111111 |

Una diferencia produce `E0ParityError`; no se ajustan parámetros para forzar
el resultado.

## Siguiente bloque

La ablación con BGE-large es el siguiente bloque previsto, pero **no está
implementada** aquí.

