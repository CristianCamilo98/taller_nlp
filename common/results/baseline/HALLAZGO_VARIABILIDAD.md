# Hallazgo: variabilidad del LLM entre ejecuciones

## Observación

Ejecutando el mismo agente, con el mismo modelo (Gemini 3.8 Flash,
temperature=0), el mismo system prompt y el mismo golden set (20 preguntas),
los resultados específicos por pregunta cambian entre ejecuciones.

## Evidencia

Comparativa baseline v1 vs baseline v2:

| ID | v1 | v2 |
|---|---|---|
| g3-008 (NVDA export controls) | acierto_cita ✅ | acierto_cita ❌ |
| g3-011 (META privacidad) | acierto_cita ❌ | acierto_cita ✅ |

**Métricas globales idénticas:** cifra 100%, cita 66.7%, trayectoria 100%.
**Aciertos específicos distintos.**

## Causas

1. **Tool calling no determinista.** El modelo, incluso con temperature=0,
   no siempre elige el mismo chunk entre los k mejores.
2. **Búsqueda densa con empates.** Cuando dos chunks tienen similitud
   parecida, el orden puede variar ligeramente.
3. **Longitud del contexto.** El agente acumula tool calls; pequeñas
   diferencias en el orden pueden llevar a respuestas finales distintas.

## Implicaciones

1. **La métrica global es estable** (4/6 siempre) → es fiable para comparar.
2. **La métrica por pregunta no es determinista** → no usar una sola
   ejecución para juzgar si una pregunta concreta "pasa o falla".
3. **Para mayor robustez**, se podría ejecutar el golden set N veces
   (3-5) y promediar aciertos por pregunta.

## Recomendación para el informe

Reportar la métrica global como resultado principal. En la presentación,
mencionar la variabilidad como limitación y como posible línea de mejora
(ejecuciones múltiples y promedio).