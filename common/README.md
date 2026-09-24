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

### Perfiles de retrieval denso

El perfil por defecto sigue siendo el baseline BGE y no requiere ninguna
variable nueva:

```powershell
$env:MIAX_RETRIEVAL_PROFILE = 'bge-small-baseline'
```

El perfil Qwen se selecciona explícitamente con:

```powershell
$env:MIAX_RETRIEVAL_PROFILE = 'qwen3-06b'
```

El perfil Gemini Embedding 2 usa OpenRouter para codificar únicamente las
queries y reutiliza su índice persistido independiente:

```powershell
$env:MIAX_RETRIEVAL_PROFILE = 'gemini-embedding-2'
$env:OPENROUTER_API_KEY = '<clave>'
```

El directorio `indice_faiss_gemini_embedding_2/` debe contener
`corpus.faiss`, `chunks_meta.parquet` e `index_manifest.json`. El runtime
valida antes de llamar a OpenRouter: modelo gestionado por proveedor,
dimensión 3072, normalización L2, `IndexFlatIP`, 1.749 vectores y hashes de
FAISS/corpus/metadata. Las queries se envían raw solicitando
`input_type=search_query`. Si OpenRouter rechaza ese campo con HTTP 400, el
fallback sin `input_type` emite un warning y queda disponible como
`input_type_mode=fallback_without_input_type` en la provenance de la última
query; nunca ocurre silenciosamente. El índice histórico solo demuestra los
`input_type` solicitados, por lo que su manifest declara honestamente
`input_type_historical_effective=unknown`.

Qwen resuelve un índice separado en `indice_faiss_qwen3_06b/`. Ese directorio
debe contener `corpus.faiss`, `chunks_meta.parquet` e `index_manifest.json`.
Si ese bundle ya existe, basta seleccionar el perfil; el runtime valida su
manifest y sus hashes antes de cargar Qwen:

```powershell
$env:MIAX_RETRIEVAL_PROFILE = 'qwen3-06b'
```

En un clon limpio con el dataset oficial original, el bundle se construye con:

```powershell
$env:MIAX_DATASET_DIR = 'C:\ruta\al\dataset'
$env:HF_HUB_OFFLINE = '1'
python -m common.scripts.build_qwen_index
```

El modelo exacto debe estar previamente disponible en la caché local. El
builder usa `local_files_only=True`: no descarga modelos ni sobrescribe un
bundle existente. Codifica los 1.749 documentos raw en su orden oficial,
reafirma L2 en float32, crea `IndexFlatIP`, copia sin alterar el metadata
canónico y registra los hashes reales. El manifest tiene este contrato:

```json
{
  "schema_version": 1,
  "profile": "qwen3-06b",
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
  "embedding_dimension": 1024,
  "pooling": "lasttoken",
  "include_prompt": true,
  "normalize": true,
  "document_format": "raw",
  "query_template": "Instruct: Given a financial question, retrieve relevant passages from SEC 10-K filings that answer the question.\nQuery:{query}",
  "index_type": "IndexFlatIP",
  "ntotal": 1749,
  "faiss_sha256": "<SHA-256 real del corpus.faiss generado>",
  "chunks_sha256": "388ff3671742c2248e8f5cb1c75afbc786edfa0dc6f72a62d1fadf2310ec82b2",
  "chunks_meta_sha256": "fbd22360e517e5da250b18f0c95dbc9ad704b41902cedb1154747fde0be16fd9"
}
```

El SHA `f2180a18b804b1e767e34d1ad84bed3cb3af07c9206e488f3fd6d3cfd872d2de`
identifica el índice congelado usado para validar esta integración. Una
reconstrucción reproducible puede tener otro SHA por diferencias numéricas
entre hardware, pero su archivo debe coincidir con el SHA que declara su propio
manifest. Modelo, revisión, pooling, dimensión, normalización, formato de
documentos, hashes de corpus/metadata, tipo de índice y `ntotal` no se relajan.
El runtime falla antes de cargar Qwen si cualquiera de ellos no coincide.

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

### Benchmark de retrieval v2 (48 preguntas)

El runner de COMMON usa el benchmark congelado por EvidenceSpan y permite
seleccionar una pregunta para smoke. Este comando realiza exactamente una
query Gemini, ranking global FAISS y postfiltrado por ticker/año/item:

```powershell
python -m common.scripts.evaluate_retrieval_benchmark --profile gemini-embedding-2 --question-id dani-b6-001 --output common\results\retrieval\gemini_smoke_dani-b6-001.json
```

Sin `--question-id`, el mismo runner evalúa las 48 preguntas. El JSON conserva
hashes de benchmark, índice y metadata, ranking, métricas y provenance de la
petición de embeddings. No incluye rewriting, BM25 ni reranking.
