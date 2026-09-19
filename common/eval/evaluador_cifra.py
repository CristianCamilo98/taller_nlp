"""Evaluador numérico estricto, estructurado y resistente a substrings."""

from __future__ import annotations

import math

from common.xbrl import find_xbrl_fact

REL_TOLERANCE = 0.005
ABS_TOLERANCE = 1e-9
PERCENTAGE_ABS_TOLERANCE = 0.1


def _value(result: dict, name: str):
    return result.get(name, result.get(f"{name}_agente"))


def _matches(actual, expected, *, rel=REL_TOLERANCE,
             absolute=ABS_TOLERANCE) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=rel,
                            abs_tol=absolute)
    except (TypeError, ValueError):
        return False


def _tool_calls(result: dict) -> list[dict]:
    return result.get("tool_calls_detallado") or []


def _has_xbrl_call(result: dict, ticker: str, year: int, concept: str) -> bool:
    for call in _tool_calls(result):
        if call.get("name") != "get_xbrl_fact":
            continue
        args = call.get("args") or {}
        try:
            called_year = int(args.get("fiscal_year"))
        except (TypeError, ValueError):
            continue
        if (str(args.get("ticker", "")).strip().upper() == ticker.upper()
                and called_year == int(year)
                and args.get("concept") == concept):
            return True
    return False


def _golden_fact(ticker: str, year: int, concept: str):
    rows = find_xbrl_fact(ticker, year, concept)
    return None if rows.empty else rows.iloc[0]


def evaluar_cifra_detallada(resultado: dict, pregunta: dict) -> dict:
    family = pregunta.get("familia")
    if family not in {"numerica", "comparativa"}:
        return {"aplica": False, "acierto_cifra": None, "errores": []}

    errors: list[str] = []
    ticker = str(pregunta.get("ticker") or "").upper()
    concept = pregunta.get("concept_esperado") or pregunta.get("concept_xbrl")
    unit = pregunta.get("unidad")

    if str(_value(resultado, "ticker") or "").upper() != ticker:
        errors.append("ticker")
    if _value(resultado, "fuente") not in {"xbrl", "ambas"}:
        errors.append("fuente_no_xbrl")
    if _value(resultado, "unidad") != unit:
        errors.append("unidad")
    response_concept = _value(resultado, "concepto_xbrl")
    if response_concept != concept:
        errors.append("concepto_respuesta")

    if family == "numerica":
        year = int(pregunta["fiscal_year"])
        expected = pregunta.get("cifra_esperada")
        row = _golden_fact(ticker, year, concept)
        if row is None or not _matches(float(row["value"]), expected,
                                       rel=0.0, absolute=ABS_TOLERANCE):
            errors.append("golden_no_coincide_xbrl")
        if _value(resultado, "ejercicio") != year:
            errors.append("ejercicio")
        if not _matches(_value(resultado, "cifra"), expected):
            errors.append("valor")
        if not _has_xbrl_call(resultado, ticker, year, concept):
            errors.append("llamada_xbrl")
    else:
        start = int(pregunta["fiscal_year_inicio"])
        end = int(pregunta["fiscal_year_fin"])
        initial = pregunta["valor_inicial_esperado"]
        final = pregunta["valor_final_esperado"]
        delta = pregunta["delta_esperado"]
        percentage = pregunta.get("porcentaje_esperado")
        first = _golden_fact(ticker, start, concept)
        last = _golden_fact(ticker, end, concept)
        if (first is None or last is None
                or not _matches(float(first["value"]), initial,
                                rel=0.0, absolute=ABS_TOLERANCE)
                or not _matches(float(last["value"]), final,
                                rel=0.0, absolute=ABS_TOLERANCE)):
            errors.append("golden_no_coincide_xbrl")
        expected_fields = {
            "ejercicio_inicial": start,
            "ejercicio_final": end,
            "valor_inicial": initial,
            "valor_final": final,
            "delta": delta,
            "cifra": delta,
        }
        for name, expected in expected_fields.items():
            actual = _value(resultado, name)
            if name.startswith("ejercicio"):
                if actual != expected:
                    errors.append(name)
            elif not _matches(actual, expected):
                errors.append(name)
        if percentage is not None and not _matches(
                _value(resultado, "porcentaje"), percentage,
                rel=0.0, absolute=PERCENTAGE_ABS_TOLERANCE):
            errors.append("porcentaje")
        if not _has_xbrl_call(resultado, ticker, start, concept):
            errors.append("llamada_xbrl_inicio")
        if not _has_xbrl_call(resultado, ticker, end, concept):
            errors.append("llamada_xbrl_fin")

    return {"aplica": True, "acierto_cifra": not errors, "errores": errors}


def evaluar_cifra(resultado: dict, pregunta: dict) -> bool | None:
    return evaluar_cifra_detallada(resultado, pregunta)["acierto_cifra"]
