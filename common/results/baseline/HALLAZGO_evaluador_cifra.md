# Hallazgo: desajuste entre evaluador de cifra y diseño del golden set

## Problema

El evaluador de cifra original comparaba `cifra_agente` con `cifra_esperada`
usando igualdad numérica. Para las preguntas comparativas del tipo
"¿Cuánto creció X entre FY2024 y FY2025?", el golden set guarda como
`cifra_esperada` el **valor absoluto de FY2025** (por convención del
enunciado), pero el agente responde con el **delta** (diferencia entre
los dos años), que es lo que la pregunta pide literalmente.

Resultado: 0/7 aciertos en comparativas numéricas, cuando el agente
realmente responde correctamente a lo que se le pregunta.

## Ejemplos

| Pregunta | cifra_esperada | cifra_agente | Veredicto original | Veredicto humano |
|---|---|---|---|---|
| MSFT revenue FY2024→FY2025 | 281.724M | 36.602M (delta) | ❌ | ✅ |
| NVDA revenue FY2024→FY2025 | 130.497M | 69.575M (delta) | ❌ | ✅ |
| META I+D FY2024→FY2025 | 57.372M | 13.499M (delta) | ❌ | ✅ |

## Solución

Se reimplementa `evaluador_cifra.py` con tres reglas de aceptación:

1. **Coincidencia numérica** con `cifra_esperada` (caso original).
2. **Cifra esperada presente en la prosa** de la respuesta (formato flexible).
3. **Para comparativas:** aceptar también el delta `cifra_esperada - valor_FY_anterior`.

Con esto, las comparativas correctas se cuentan como acierto.

## Decisión de diseño del golden set

Mantenemos el formato del enunciado (cifra_esperada = valor absoluto
de FY2025) para ser comparables con el golden oficial del profesor.
El evaluador se adapta para reconocer ambas formas de responder.

## Impacto

- `acierto_cifra` sube de 42.9% a X% (rellenar tras ejecutar).
- No cambia el comportamiento del agente. Solo cambia el criterio de evaluación.