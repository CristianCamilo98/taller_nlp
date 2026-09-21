# Experimentos de Cristian

Punto de partida: tag `baseline-comun-v2`.
`common/` no se modifica salvo bug compartido.

## Convención

| Ruta | Uso |
| --- | --- |
| `common/` | Baseline congelado (tools, agente, eval, `responder`/`evaluar`) |
| `cristian/experiments/` | Mejoras y ablations de este experimento |
| `cristian/experiments/results/` | Métricas JSON regenerables |
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
| `agent.py` | Agente experimental (OpenRouter + tools locales). |
| `tools.py` | Las 4 tools (contrato de la práctica); `search_filings` → `miax_s1`. |
| `schema.py` / `config.py` / `xbrl.py` | Copia experimental autónoma (no importa `common`). |
| `../../.env` | Claves locales (`OPENROUTER_API_KEY`, `HF_TOKEN`, `MIAX_DATASET_DIR`). |

```bash
cp .env.example .env   # si aún no existe
# edita .env y pega tus tokens
```

## Hipótesis (por definir)

_Pendiente: qué eje se mejora primero (routing de tools, retrieval, guardrail, prompts, …)._
