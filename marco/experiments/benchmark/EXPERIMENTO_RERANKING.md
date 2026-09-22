# Experimento de reranking: MiniLM vs BGE-reranker-v2-m3

## Arquitectura (ablación pura)
- Candidate generator: `BAAI/bge-small-en-v1.5` (idéntico al baseline común).
- Chunks e índice: originales.
- Candidate pool: top-20 densos.
- R0: `cross-encoder/ms-marco-MiniLM-L-6-v2` (rev `233902d2...`).
- R1: `BAAI/bge-reranker-v2-m3` (rev `953dc6f6...`).
- Top-k final: 20.

## Resultados (48 preguntas, benchmark FROZEN)

### Vista principal: NON-7A-36 R@5 (métrica prefijada)
| Config | R@5 | MRR@10 | R@1 |
|---|---|---|---|
| Dense | 0.5278 | 0.3607 | 0.2222 |
| R0 MiniLM | 0.5556 | 0.3323 | 0.1944 |
| **R1 BGE** | **0.7500** | **0.6027** | **0.4722** |
| Δ R1-Dense | **+22.22pp** | **+24.20pp** | **+25.00pp** |

### ALL-48
| Config | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|
| Dense | 0.2708 | 0.6458 | 0.7917 | 0.4264 |
| R0 MiniLM | 0.2708 | 0.6667 | 0.7708 | 0.4180 |
| **R1 BGE** | **0.5833** | **0.8125** | **0.8750** | **0.6882** |

### Por Item (R@5)
| Item | Dense | R0 MiniLM | R1 BGE | Δ R1-Dense |
|---|---|---|---|---|
| ITEM-1A | 0.5833 | 0.6667 | 0.6667 | +8.33pp |
| ITEM-7 | 0.5833 | 0.6667 | 0.9167 | +33.33pp |
| ITEM-7A | 1.0000 | 1.0000 | 1.0000 | 0 |
| ITEM-8 | 0.4167 | 0.3333 | 0.6667 | +25.00pp |

## Truncamiento
- 7/48 chunks (14.58%) superan los 512 tokens de MiniLM.
- R1 resuelve 5/7 de los truncados (8192 tokens de contexto).
- Los 2 restantes no son problema de truncamiento, sino de candidate recall@20 (66.7% en ITEM-8).

## Latencia
| Config | Latencia media |
|---|---|
| Dense | 194.7 ms |
| R0 MiniLM | 434.6 ms |
| R1 BGE | 7889.3 ms |

## Conclusión
MiniLM-L6 (R0) no mejora el baseline denso. BGE-reranker-v2-m3 (R1) ofrece una
mejora sustancial (+22.22pp R@5, +24.20pp MRR en NON-7A-36), y rompe el cuello
de botella de ITEM-8 (0.4167 → 0.6667 en R@5). El coste es un aumento de
latencia de ~40x, justificable en escenarios donde la calidad prima sobre la
latencia.

## Recomendación
Integrar BGE-reranker-v2-m3 como R1 del pipeline de retrieval.
Explorar cuantización/destilación como trabajo futuro para reducir latencia.