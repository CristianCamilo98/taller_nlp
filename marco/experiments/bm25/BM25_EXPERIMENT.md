# Experimento BM25 + dense (ablación causal B0 vs B1)

## Base
- Rama: `experiment/marco-bm25-hybrid`
- Ancestro: `baseline-comun-v2` (`83f80393ab1d13b3b0b42369a7adeb3134337530`)
- Benchmark: `dani/experiments/benchmark/retrieval_benchmark_v2.jsonl`
- Benchmark SHA-256: `6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C`
- Chunks SHA-256: `388FF3671742C2248E8F5CB1C75AFBC786EDFA0DC6F72A62D1FADF2310EC82B2`
- Chunks_meta SHA-256: `FBD22360E517E5DA250B18F0C95DBC9AD704B41902CEDB1154747FDE0BE16FD9`
- FAISS SHA-256: `605CE725DE4604A3CD37A35EE61A03D52AA9351543D8CA98F53D5535DC9E8F5A`

## Configuración (congelada antes de ver resultados)
- Embedding: `BAAI/bge-small-en-v1.5`
- Índice: `IndexFlatIP` sobre 1749 chunks
- Política metadata: ranking global → postfilter
- Candidate pool por retriever: 20
- BM25: `BM25Okapi(k1=1.5, b=0.75)`
- Tokenización: `[^\W_]+` sobre lowercase, tokens ≥2 chars, sin stopwords
- RRF: `k_rrf=60`, ranks 1-based
- Tie-break RRF: (-rrf, dense_rank, bm25_rank, chunk_id)

## B0 vs B1
- **B0:** dense BGE-small
- **B1:** dense + BM25 + RRF

## Métricas
- R@1, R@3, R@5, R@10, MRR@10
- Vistas: ALL-48, NON-7A-36, ITEM-1A-12, ITEM-7-12, ITEM-7A-12, ITEM-8-12
- Principal: NON-7A R@5
- Secundaria: NON-7A MRR@10

## Hipótesis explicativas plausibles

Las siguientes son hipótesis, no causas demostradas. No se ha realizado
ablación causal específica de idioma, pesos o postfilter. Se dejan como
explicaciones plausibles que justifican por qué BM25+RRF no aporta en
este corpus concreto:

1. **Idioma cruzado:** las queries están en español, el corpus en inglés.
   BM25 puede no encontrar coincidencias léxicas relevantes.
2. **RRF asume rankers comparables:** con k_rrf=60, el top-1 de cada
   método pesa igual. Si un método es sistemáticamente peor, contamina
   la fusión.
3. **Postfilter sobre ranking global:** el comportamiento de BM25 en
   ranking global no se ha aislado del filtro posterior.

Para convertir estas hipótesis en causas habría que ejecutar ablaciones
específicas (traducción, pesos, prefilter). No se ha hecho por límite de
tiempo.

## Restricciones
- Sin LLM, sin API, sin reranker, sin Qwen
- common/ idéntico a baseline-comun-v2
- Fail-closed: las 48 preguntas siempre en denominador

## Ejecución
```bash
python marco/experiments/bm25/runner_bm25.py