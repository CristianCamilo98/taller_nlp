# Mejora 2: Guardrail XBRL

## Resultado

| Métrica | Baseline v2 | Final v3 | Δ |
|---|---|---|---|
| acierto_cifra | 100% | 100% | 0pp |
| acierto_cita | 66.7% | **83.3%** | **+16.7pp** |
| acierto_trayectoria | 100% | 100% | 0pp |
| latencia media | 11.64s | 25.02s | **+13.4s** |

## Efecto inesperado

El guardrail no solo verifica las cifras: **también mejora el retrieval**.
Al detectar que el agente necesita verificar, este reinicia la búsqueda y
encuentra el chunk correcto para alguna pregunta que antes fallaba.

## Coste: latencia

La latencia media sube 13.4s. **Un caso concreto dispara la media:**
g3-013 tarda 215.9s porque el guardrail provoca reintentos en una pregunta
que es intrínsecamente difícil (ver "Hallazgo g3-013").

Sin ese caso, la latencia media sería:
(25.02 × 20 - 215.9 + 41.8) / 20 = **~16.3s**, mucho más razonable.

## Hallazgo: g3-013 es una pregunta mal diseñada

**Pregunta:** "¿Qué políticas contables relevantes describe Amazon en sus
estados financieros de FY2024?"

**Ancla esperada:** "We utilize a two-step approach to recognizing and
measuring uncertain income tax positions (income tax contingencies)."

**Problema:** la pregunta pide "políticas contables relevantes" (ambiguo),
pero el ancla apunta a una política concreta (income tax). El agente
responde con revenue recognition (también correcto), pero no coincide con
el ancla.

**Conclusión:** es un fallo del diseño del golden set, no del agente.
Se documenta como limitación y como ejemplo de la dificultad de definir
"la respuesta correcta" en preguntas abiertas.

## Decisión

El guardrail se queda. La mejora en acierto_cita compensa el coste en
latencia. Se añade `ToolCallLimitMiddleware` para cortar bucles en
preguntas difíciles.