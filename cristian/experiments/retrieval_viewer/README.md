# Visor de retrieval (benchmark v2)

Compara JSON de Dani / Marco / Cristian en
`cristian/experiments/results/comparisons/`.

```bash
# Generar Gemini + BGE-reranker y copiar al directorio del viewer
python -m cristian.experiments.eval_retrieval_benchmark_v2 --rerank \
  --copy-to-comparisons

python -m cristian.experiments.retrieval_viewer
```

Los JSON con `metrics.dense` / `metrics.reranked` (Marco o Cristian+rerank)
aparecen como **dos runs** en el dashboard.

Métrica primaria: **NON-7A-36 Recall@5** (la que fija Dani para el agente con `k=5`).
