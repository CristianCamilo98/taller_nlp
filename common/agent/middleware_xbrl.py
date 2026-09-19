"""Guardrail XBRL basado solo en las tools usadas durante la ejecución."""

from __future__ import annotations

import math

from common.xbrl import find_xbrl_fact

REL_TOLERANCE = 0.005
ABS_TOLERANCE = 1e-9
PERCENTAGE_ABS_TOLERANCE = 0.1


def numbers_match(actual, expected, *, rel=REL_TOLERANCE,
                  absolute=ABS_TOLERANCE) -> bool:
    try:
        return math.isclose(float(actual), float(expected),
                            rel_tol=rel, abs_tol=absolute)
    except (TypeError, ValueError):
        return False


def _queried_facts(tool_calls: list[dict]) -> list[dict]:
    facts = []
    for call in tool_calls:
        if call.get("name") != "get_xbrl_fact":
            continue
        args = call.get("args") or {}
        ticker = str(args.get("ticker") or "").strip().upper()
        concept = args.get("concept")
        try:
            year = int(args.get("fiscal_year"))
        except (TypeError, ValueError):
            continue
        if not ticker or not concept:
            continue
        rows = find_xbrl_fact(ticker, year, str(concept))
        if rows.empty:
            continue
        row = rows.iloc[0]
        facts.append({
            "ticker": ticker,
            "fiscal_year": year,
            "concept": str(concept),
            "unit": str(row["unit"]),
            "value": float(row["value"]),
        })
    return facts


def verificar_respuesta_xbrl(
    respuesta: dict, tool_calls: list[dict] | None = None,
) -> tuple[bool, str]:
    """Comprueba la respuesta contra hechos XBRL realmente consultados.

    No conoce el golden ni decide si el concepto consultado responde a la
    pregunta. Esa correspondencia pertenece al evaluador de trayectoria.
    """
    numeric_fields = ("cifra", "valor_inicial", "valor_final", "delta",
                      "porcentaje")
    if not any(respuesta.get(field) is not None for field in numeric_fields):
        return True, "sin cifra que verificar"
    if respuesta.get("fuente") not in {"xbrl", "ambas"}:
        return False, "Toda cifra debe declarar fuente XBRL o ambas."

    consulted = _queried_facts(tool_calls or [])
    if not consulted:
        return False, "No hay una llamada get_xbrl_fact válida en la ejecución."

    ticker = str(respuesta.get("ticker") or "").strip().upper()
    concept = respuesta.get("concepto_xbrl")
    unit = respuesta.get("unidad")
    start = respuesta.get("ejercicio_inicial")
    end = respuesta.get("ejercicio_final")

    def matching(year):
        return [fact for fact in consulted
                if fact["ticker"] == ticker
                and fact["fiscal_year"] == int(year)
                and fact["concept"] == concept]

    if start is not None or end is not None:
        if start is None or end is None:
            return False, "La comparación debe identificar ambos ejercicios."
        first_candidates = matching(start)
        last_candidates = matching(end)
        if not first_candidates or not last_candidates:
            return False, ("Faltan las dos llamadas XBRL del ticker, concepto "
                           "y ejercicios declarados.")
        first, last = first_candidates[0], last_candidates[0]
        if first["unit"] != unit or last["unit"] != unit:
            return False, "La unidad no coincide con ambos hechos consultados."
        initial, final = first["value"], last["value"]
        delta = final - initial
        percentage = (delta / initial * 100.0) if initial else None
        checks = [
            ("valor_inicial", respuesta.get("valor_inicial"), initial,
             REL_TOLERANCE, ABS_TOLERANCE),
            ("valor_final", respuesta.get("valor_final"), final,
             REL_TOLERANCE, ABS_TOLERANCE),
            ("delta", respuesta.get("delta"), delta,
             REL_TOLERANCE, ABS_TOLERANCE),
            ("cifra", respuesta.get("cifra"), delta,
             REL_TOLERANCE, ABS_TOLERANCE),
        ]
        if percentage is not None:
            checks.append(("porcentaje", respuesta.get("porcentaje"),
                           percentage, 0.0, PERCENTAGE_ABS_TOLERANCE))
        for name, actual, expected, rel, absolute in checks:
            if not numbers_match(actual, expected, rel=rel, absolute=absolute):
                return False, f"{name}={actual!r} no coincide con {expected!r}."
        return True, "comparación respaldada por dos llamadas XBRL"

    year = respuesta.get("ejercicio")
    if year is None or not concept:
        return False, "La respuesta numérica no identifica año y concepto."
    candidates = matching(year)
    if not candidates:
        return False, ("La respuesta no corresponde a un hecho XBRL consultado "
                       "con el mismo ticker, año y concepto.")
    for fact in candidates:
        if (fact["unit"] == unit
                and numbers_match(respuesta.get("cifra"), fact["value"])):
            return True, "respuesta respaldada por la llamada XBRL"
    return False, "Unidad o valor distintos del hecho XBRL consultado."


def verificar_cifra(respuesta: dict, ticker: str | None = None,
                    fiscal_year: int | None = None,
                    concept: str | None = None,
                    unidad: str | None = None,
                    tool_calls: list[dict] | None = None) -> tuple[bool, str]:
    """Compatibilidad con el nombre anterior."""
    candidate = dict(respuesta)
    if ticker is not None:
        candidate.setdefault("ticker", ticker)
    if fiscal_year is not None:
        candidate.setdefault("ejercicio", fiscal_year)
    if concept is not None:
        candidate.setdefault("concepto_xbrl", concept)
    if unidad is not None:
        candidate.setdefault("unidad", unidad)
    calls = tool_calls if tool_calls is not None else (
        candidate.get("tool_calls_detallado") or [])
    return verificar_respuesta_xbrl(candidate, calls)
