# Visor de resultados

Dashboard HTML local para comparar runs de evaluación (`*.jsonl` en
`cristian/experiments/results/`). No toca `common/` ni lee `.env`.

## Uso

Desde la raíz del repo:

```bash
python -m cristian.experiments.results_viewer
```

Eso:

1. Lista automáticamente todos los `*.jsonl` de `results/`
2. Agrupa por `run_id` (varios runs en el mismo fichero, o un fichero por run)
3. Escribe `dashboard.html` (gitignored) y lo abre en el navegador

Opciones:

```bash
python -m cristian.experiments.results_viewer --no-open
python -m cristian.experiments.results_viewer --out /tmp/cmp.html
python -m cristian.experiments.results_viewer --results-dir cristian/experiments/results
```

Cada vez que generes un nuevo JSONL con `evaluar`, vuelve a ejecutar el mismo
comando: el dashboard se regenera desde disco (sin datos inventados).

## Qué muestra

- **Resumen por run**: % acierto cifra / cita / trayectoria (solo donde aplica),
  latencia media, coste total/medio, tool calls medios, errores, prompt, commit.
- **Desglose por familia** (`numerica` / `extractiva` / `comparativa`).
- **Heatmap pregunta × run** (✓ / ✗ / —) con selector de métrica.
- **Drill-down**: clic en una fila → side-by-side `respuesta_agente` vs esperada,
  tools, errores de métricas.

## Nota sobre Cursor Canvas

Canvas no puede leer ficheros del repo (solo datos embebidos). Este visor es la
vía reejecutable; regenera el HTML cuando haya nuevos resultados.
