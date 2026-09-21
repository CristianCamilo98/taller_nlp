# Intento fallido: reranking con cross-encoder

## Hipótesis
Rerankear top-20 con cross-encoder local podría mejorar el retrieval.

## Resultado (48 preguntas, benchmark Dani v2 FROZEN)

| Métrica | Densa | Reranked | Δ |
|---|---|---|---|
| R@1 | 27.08% | 27.08% | 0 |
| R@3 | 52.08% | 47.92% | −4.16pp |
| R@5 | 64.58% | 66.67% | +2.08pp |
| R@10 | 79.17% | 77.08% | −2.09pp |
| MRR | 0.426 | 0.418 | −0.008 |

## Conclusión
No mejora. En 6 preguntas parecía mejorar (+16.7pp R@3), pero era artefacto
de muestra pequeña. Sobre 48 preguntas, es neutral o peor.

## Alternativa que sí funciona
Qwen3-0.6B (Dani): R@1 27% → 45.8%, R@5 64.6% → 81.25%. Esa es la mejora real.