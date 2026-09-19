## Baseline común v2 (en preparación)

Runtime compartido para las cuatro herramientas y el agente de informes 10-K.
Se recomienda Python 3.11 o 3.12. El import de `common` y de sus interfaces no
carga el dataset, descarga embeddings ni crea clientes de red.

### Dataset

La resolución es única y no depende del directorio de trabajo. Puede definirse:

```powershell
$env:MIAX_DATASET_DIR = 'C:\ruta\al\dataset'
```

Si no existe esa variable se prueban, en este orden, `repo/dataset` y
`parent_del_repo/dataset`. Dentro deben existir:

- `corpus_miax_2026/{secciones.jsonl,chunks.jsonl,xbrl_facts.parquet}`
- `indice_faiss/{corpus.faiss,chunks_meta.parquet}`

Una variable definida pero incorrecta produce un error explícito; no se cae
silenciosamente en otra copia del dataset.

### Qué forma parte del baseline

- Runtime: `agent/`, `tools/`, `eval/`, `responder.py`, `evaluar.py`,
  `config.py`, `benchmark_config.py` y `xbrl.py`.
- Referencia docente: `starter/`. Conserva el retrieval denso BGE/FAISS dado.
- Histórico, no runtime: `results/` y `notebooks/`.
- Experimento, no runtime: `retrieval/query_rewriting.py` y los directorios
  personales bajo `*/experiments/`.

Ningún módulo runtime importa query rewriting, resultados ni scripts
históricos. El baseline de retrieval sigue siendo BGE small EN v1.5, vectores
normalizados, `IndexFlatIP`, ranking sobre 1.749 vectores y postfiltrado por
metadata. No incluye BM25, reranking ni cambios de chunking/embeddings.

### Configuración congelada

`benchmark_config.py` centraliza modelo, proveedor, temperatura, límites,
reintentos, k y versión del prompt. El modelo solicitado sigue siendo
`openrouter:google/gemini-3.8-flash`; la temperatura pasa a estar declarada
explícitamente como `0.0`. El máximo de 3 búsquedas del prompt es una guía
heredada y no se puntúa en trayectoria porque el enunciado no fija ese máximo.

Las métricas de proveedor que no estén presentes en la respuesta (`coste`,
modelo efectivo o tokens) se guardan como `null`; no se estiman. La latencia de
pared incluye backoff, `latencia_activa_s` lo separa y la pausa entre preguntas
queda fuera de ambas.

### Instalación y tests

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r common\requirements.txt
python -m unittest discover -s common\tests -v
```

Recall@1/3/5/10 y MRR@10 offline (requiere que el modelo de embeddings ya esté en caché o que el
entorno permita descargarlo; el script no llama a un LLM):

```powershell
python -m common.scripts.measure_recall --output recall-baseline-v2.json
```

El resultado canónico debe registrar commit, hash del golden y hashes de los
cinco artefactos del dataset. No debe compararse directamente con el baseline
histórico: v2 cambia el golden comparativo y endurece los tres evaluadores.
