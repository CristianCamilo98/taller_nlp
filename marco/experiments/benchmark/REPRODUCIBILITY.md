# Reproducibilidad — Experimento de reranking

## Rama y ascendencia
- Rama: `experiment/marco-reranking`
- Ancestro: `baseline-comun-v2` (`83f8039`)

## Benchmark
- Fichero: `marco/experiments/benchmark/data/retrieval_benchmark_v2.jsonl`
- Preguntas: 48
- SHA-256: `6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C`
- Origen: `dani/experiments/benchmark/retrieval_benchmark_v2.jsonl`
  (rama `experiment/dani-embeddings-chunking`, tag `dani-retrieval-benchmark-v2`)

## Dataset
| Fichero | SHA-256 |
|---|---|
| `dataset/corpus_miax_2026/chunks.jsonl` | `388FF3671742C2248E8F5CB1C75AFBC786EDFA0DC6F72A62D1FADF2310EC82B2` |
| `dataset/indice_faiss/corpus.faiss` | `605CE725DE4604A3CD37A35EE61A03D52AA9351543D8CA98F53D5535DC9E8F5A` |
| `dataset/indice_faiss/chunks_meta.parquet` | `FBD22360E517E5DA250B18F0C95DBC9AD704B41902CEDB1154747FDE0BE16FD9` |

## Modelos
| Rol | Modelo | Revision |
|---|---|---|
| Embedding (candidate generator) | `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (verificar) |
| Reranker R0 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |

## Entorno
- Python: `3.14.3` (`.venv/Scripts/python.exe`)
- Nota: Python 3.14 emite warnings de Pydantic V1 en LangChain. Funciona pero es frágil.

## Arquitectura del experimento (ablación pura)
1. Candidate generator: `BAAI/bge-small-en-v1.5` (idéntico al baseline común).
2. Chunks e índice: originales entregados.
3. Candidate pool: top-20 densos.
4. Reranker: reordena top-20.
5. Devuelve: top-k.

## Protocolo de evaluación
- Ground truth: EvidenceSpan `[char_start, char_end)` mapeado por contención completa usando `inicio_car/fin_car` del chunk.
- Fail-closed: si alguna de las 48 no mapea, el runner falla.
- Métricas: Recall@1/3/5/10, MRR@10.
- Adicional para reranking: Candidate Recall@20.
- Vistas: ALL-48, ITEM-1A-12, ITEM-7-12, ITEM-7A-12, ITEM-8-12, NON-7A-36.

## Comando de reproducción
```bash
python marco/experiments/benchmark/runner_canonico.py