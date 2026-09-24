# Experimentos de Cristian

Punto de partida: tag `baseline-comun-v2`.
`common/` no se modifica salvo bug compartido.

## Convención

| Ruta | Uso |
| --- | --- |
| `common/` | Baseline congelado (tools, agente, eval, `responder`/`evaluar`) |
| `cristian/experiments/` | Mejoras y ablations de este experimento |
| `cristian/experiments/results/` | Métricas JSON regenerables |
| `cristian/experiments/results_viewer/` | Dashboard HTML para comparar runs |
| `cristian/experiments/artifacts/` | Índices/cachés locales (gitignored) |

## Cómo comparar

1. Ejecutar métricas del baseline vía `common` (sin cambios).
2. Implementar la mejora aquí.
3. Medir el mismo golden / mismas métricas.
4. Guardar resultados en `results/` con nombre del experimento.

## Arranque

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r cristian/experiments/requirements.txt

export MIAX_DATASET_DIR="/ruta/a/cristian/dataset"   # ya está en .env

python -m unittest discover -s common/tests -v
```

## Ficheros de trabajo

| Fichero | Rol |
| --- | --- |
| `miax_s1.py` | Retrieval denso (FAISS + BGE). Punto de partida a mejorar. |
| `agent.py` | Agente experimental + CLI interactiva. |
| `responder.py` | `responder(pregunta)` → dict evaluable. |
| `evaluar.py` | Corre golden + métricas cifra/cita/trayectoria. |
| `eval_retrieval_benchmark_v2.py` | Retrieval vs benchmark Dani v2 (Recall@k / MRR). |
| `reranker.py` | Cross-encoder `BAAI/bge-reranker-v2-m3` sobre pool denso. |
| `benchmark/` | Copia congelada de `retrieval_benchmark_v2.jsonl`. |
| `tools.py` | Las 4 tools; `search_filings` → `miax_s1`. |
| `schema.py` / `config.py` / `xbrl.py` | Runtime experimental autónomo. |
| `results_viewer/` | Dashboard HTML para comparar `results/*.jsonl` (agente). |
| `retrieval_viewer/` | Dashboard HTML para comparar retrieval benchmark v2. |
| `results/comparisons/` | JSON de Dani/Marco/Cristian para el visor de retrieval. |
| `../../.env` | Claves locales (`OPENROUTER_API_KEY`, `HF_TOKEN`, `MIAX_DATASET_DIR`). |

```bash
# Evaluación agente (golden; consume OpenRouter LLM)
python -m cristian.experiments.evaluar \
  common/golden_set/golden_set_grupo3.jsonl \
  cristian/experiments/results/run.jsonl

# Benchmark retrieval Dani v2 (sin LLM; comparable a dani/*_benchmark_v2*)
python -m cristian.experiments.eval_retrieval_benchmark_v2

# Mismo benchmark + rerank BGE (pool Gemini 20 → cross-encoder)
python -m cristian.experiments.eval_retrieval_benchmark_v2 --rerank \
  --copy-to-comparisons

# Comparar resultados agente (HTML local)
python -m cristian.experiments.results_viewer

# Comparar retrieval benchmark v2 (Dani / Marco / Cristian)
python -m cristian.experiments.retrieval_viewer
```

Los tres evaluadores se importan de `common.eval` a propósito: misma rúbrica
que el baseline, para que el delta sea comparable.

Ver también `results_viewer/README.md`.

```bash
cp .env.example .env   # si aún no existe
# edita .env y pega tus tokens
```

## Hipótesis (por definir)

_Pendiente: qué eje se mejora primero (routing de tools, retrieval, guardrail, prompts, …)._
