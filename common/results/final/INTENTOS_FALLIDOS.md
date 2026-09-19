# Intentos fallidos

## Búsqueda híbrida BM25 + densa (RRF)

**Qué se probó:** combinar búsqueda densa (BGE + FAISS) con búsqueda por
palabras clave (BM25) y fusionar rankings con Reciprocal Rank Fusion.

**Recall@k medido sobre las 6 extractivas del golden set propio:**

| k | Densa (query_es) | Densa (query_en) | Híbrida (query_en) |
|---|---|---|---|
| 1 | 1/6 | 2/6 | 2/6 |
| 3 | 2/6 | 5/6 | 4/6 |
| 5 | 4/6 | 5/6 | 4/6 |

**Por qué la híbrida no mejora:**

1. Las queries son semánticas ("¿qué riesgo describe X?"), no keyword-heavy.
   BM25 no encuentra términos técnicos que la densa no encuentre ya.
2. Con 1.749 chunks y filtros de metadatos (ticker + fiscal_year + item) ya
   aplicados, el espacio de búsqueda es pequeño y la densa acierta casi
   siempre.
3. El RRF reparte peso entre dos métodos donde uno (BM25) aporta ruido en
   lugar de señal.

**Conclusión:** se descarta. La mejora real es traducir la query al inglés.