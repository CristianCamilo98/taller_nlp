# Fallos del baseline oficial

## g3-011: META FY2024 1A — privacidad de datos

- **Ancla esperada:** "the interpretation and enforcement of the GDPR, as well as the imposition and amount of penalties for non-compliance, are subject to significant uncertainty"
- **Chunk esperado:** META-2024-1A-0057 (habla de GDPR)
- **Chunk recuperado:** META-2024-1A-0065 (habla de FTC)
- **Causa:** la búsqueda densa priorizó "privacidad regulatoria" y devolvió el primer chunk que menciona reguladores, no el que habla específicamente del GDPR.
- **Mejora esperada:** filtro por metadatos + query rewriting.

## g3-013: AMZN FY2024 8 — políticas contables

- **Ancla esperada:** "We utilize a two-step approach to recognizing and measuring uncertain income tax positions (income tax contingencies)."
- **Chunk esperado:** AMZN-2024-8-0023 (habla de income taxes)
- **Chunk recuperado:** AMZN-2024-8-0016 (habla de revenue)
- **Causa:** la query "significant accounting policies" es genérica y el corpus tiene muchas políticas contables.
- **Mejora esperada:** filtro por metadatos + query más específica.

## Conclusión

Ambos fallos son de retrieval, no del LLM ni del evaluador.
El filtro por metadatos (ticker + fiscal_year + item) debería resolverlos
porque el agente ya sabe que busca en un Item concreto (1A en g3-011,
Item 8 en g3-013).