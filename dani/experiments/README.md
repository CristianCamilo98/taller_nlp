# Experimento de retrieval de Dani — Bloques 1 a 5

## Pregunta de investigación

El trabajo estudiará embeddings y chunking mediante ablaciones controladas.
El Bloque 1 construyó la infraestructura mínima y reprodujo E0. El Bloque 2
define la evidencia sobre el texto fuente. Los Bloques 3 y 4 comparan familias
de embeddings manteniendo congelado todo el resto del retrieval.

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

El tiempo de document embedding de la primera ejecución E0 quedó contaminado
por una suspensión del equipo. No debe utilizarse como benchmark de velocidad.

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

## Ground truth independiente del chunking

Un `chunk_id` no puede ser la verdad permanente: los IDs y límites cambiarán
con cada chunker. `evidence.py` resuelve cada ancla sobre su sección original y
la representa como un intervalo de caracteres. Los offsets usan slices Python:
inicio inclusivo y final exclusivo, de modo que
`source_text[char_start:char_end]` devuelve la evidencia.

La resolución intenta primero coincidencia literal y, solo si falla, colapsa
whitespace manteniendo un mapa hacia los offsets originales. No hay fuzzy
matching. Un chunk es relevante cuando contiene por completo el span; los
overlaps parciales se registran, pero no cuentan como evidencia recuperable.

```powershell
python -B -m dani.experiments.evidence
```

El comando regenera `results/evidence_ground_truth_v1.json`. No modifica el
golden, las secciones ni los chunks comunes.

## Ablaciones de embeddings

El Bloque 3 compara BGE-small y BGE-large. El Bloque 4 añade E5-large-v2 con
su formato canónico: `query: ` para consultas y `passage: ` para documentos.
El Bloque 5 añade Qwen3-Embedding-0.6B, con su instrucción congelada solo en
la query y documentos originales. Se ejecutan offline:

```powershell
python -B -m dani.experiments.runner --experiment e1
python -B -m dani.experiments.runner --experiment e2
python -B -m dani.experiments.runner --experiment e3
```

E3 usa dimensión nativa 1024 y guarda un único resultado en
`results/e3_qwen3_embedding_06b_original.json`. No se han implementado nuevos
chunkers, BM25, reranking, query rewriting ni ablaciones MRL.

## Benchmark v2 congelado

El runner mantiene dos question sets explícitos. `--benchmark pilot` (valor
por defecto) conserva las seis preguntas y los nombres de resultado originales.
`--benchmark v2` usa las 48 preguntas de
`benchmark/retrieval_benchmark_v2.jsonl` y verifica su SHA-256 congelado antes
de cargar el modelo. Un byte distinto, o una pregunta cuyo EvidenceSpan no
quede contenido por ningún chunk, aborta la evaluación.

La relevancia v2 se deriva en cada ejecución: se filtran chunks por
ticker/ejercicio/item y se conservan todos los que cumplen
`inicio_car <= char_start` y `fin_car >= char_end`. Los IDs derivados aparecen
solo en el resultado, nunca en el benchmark. Se calculan Recall@1/3/5/10 y
MRR@10 para `ALL-48`, cada uno de los cuatro items y `NON-7A-36`.

Los resultados v2 usan el sufijo `_benchmark_v2.json`, por ejemplo
`e0_bge_small_original_benchmark_v2.json`, por lo que no colisionan con el
piloto. Un índice existente del mismo experimento, corpus y chunking puede
cargarse con `--index` solo si se aporta también, de forma explícita, el result
JSON que lo certifica mediante `--index-manifest`. Si ambos se omiten, se
conserva el comportamiento anterior de construir el índice:

```powershell
python -B -m dani.experiments.runner --experiment e0 --benchmark pilot
python -B -m dani.experiments.runner --experiment e3 --benchmark v2 `
  --index C:\ruta\al\corpus.faiss `
  --index-manifest C:\ruta\al\e3_qwen3_embedding_06b_original.json
```

La reutilización falla si no coinciden el SHA-256, tipo, dimensión o `ntotal`
del FAISS; la identidad, modelo, revisión, dimensión efectiva, normalización o
formato documental del experimento; o los hashes y número de chunks del
corpus. Las métricas y rankings del result JSON no participan en estas
decisiones de provenance.

La política cross-lingual permanece congelada: queries en español, corpus SEC
en inglés, sin traducción, query rewriting, BM25 ni reranking.
