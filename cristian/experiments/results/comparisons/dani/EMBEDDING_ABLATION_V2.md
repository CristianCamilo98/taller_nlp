# Cierre de la ablación de embeddings v2

## Diseño experimental

El objetivo de esta ablación fue comparar embeddings manteniendo constantes el corpus, el chunking original, el índice FAISS `IndexFlatIP`, la política de metadatos, el benchmark y la evaluación. La única variable experimental fue el embedding.

- Benchmark: `retrieval_benchmark_v2.jsonl`, 48 preguntas.
- SHA-256: `6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C`.
- Métrica primaria prefijada: `NON-7A Recall@5`.
- Métrica secundaria prefijada: `NON-7A MRR@10`.

## Resultados

| Experimento / modelo | ALL R@1 | ALL R@5 | ALL MRR@10 | NON-7A R@5 | NON-7A MRR@10 | 1A R@5 | 7 R@5 | 7A R@5 | 8 R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 — BGE-small | 0.270833 | 0.645833 | 0.426438 | 0.527778 | 0.360714 | 0.583333 | 0.583333 | 1.000000 | 0.416667 |
| E1 — BGE-large | 0.229167 | 0.562500 | 0.369097 | 0.416667 | 0.260648 | 0.333333 | 0.750000 | 1.000000 | 0.166667 |
| E2 — E5-large-v2 | 0.291667 | 0.770833 | 0.484896 | 0.694444 | 0.424306 | 0.750000 | 0.916667 | 1.000000 | 0.416667 |
| E3 — Qwen3-Embedding-0.6B | 0.458333 | 0.812500 | 0.597305 | 0.750000 | 0.504740 | 0.750000 | 1.000000 | 1.000000 | 0.500000 |

Item 7A alcanza `Recall@5 = 1.0` con los cuatro modelos; por ello, la vista principal prefijada es NON-7A. Item 8 sigue siendo el cuello de botella observado.

## Selección

Qwen3-Embedding-0.6B pasa a la fase de chunking porque obtiene el mayor valor observado tanto en la métrica primaria como en la secundaria. En `NON-7A Recall@5`, Qwen recupera 27/36 preguntas frente a 25/36 de E5: una diferencia de solo dos preguntas. Esta selección no implica que Qwen sea universalmente el mejor modelo.

## Caveats

- E0 no constituye una reproducción criptográfica del artefacto histórico, sino una **behavioral parity recovery using the current resolved snapshot**.
- BGE-small y BGE-large tienen 348 chunks potencialmente superiores a 512 tokens; E5 tiene 383. Esta configuración no se corrige retroactivamente porque hacerlo cambiaría la ablación congelada.
- Las 48 preguntas son un conjunto de desarrollo y selección; dejan de ser un test final independiente al usarse para escoger el embedding.
- Las 20 preguntas oficiales permanecen fuera de esta selección.

## Siguiente fase

Congelar Qwen3-Embedding-0.6B como embedding y variar únicamente el chunking.
