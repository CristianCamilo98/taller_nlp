"""Evaluador estricto de herramientas, argumentos y política común."""

from __future__ import annotations

ALLOWED = {"list_available", "get_xbrl_fact", "search_filings", "read_section"}


def _ticker(value) -> str:
    return str(value or "").strip().upper()


def _year(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _xbrl_matches(call, ticker, year, concept) -> bool:
    args = call.get("args") or {}
    return (call.get("name") == "get_xbrl_fact"
            and _ticker(args.get("ticker")) == _ticker(ticker)
            and _year(args.get("fiscal_year")) == int(year)
            and args.get("concept") == concept)


def _search_matches(call, question) -> bool:
    args = call.get("args") or {}
    try:
        positive_k = int(args.get("k", 5)) > 0
    except (TypeError, ValueError):
        positive_k = False
    return (call.get("name") == "search_filings"
            and bool(str(args.get("query") or "").strip())
            and _ticker(args.get("ticker")) == _ticker(question.get("ticker"))
            and _year(args.get("fiscal_year")) == int(question["fiscal_year"])
            and str(args.get("item")) == str(
                question.get("item", question.get("item_esperado")))
            and positive_k)


def evaluar_trayectoria_detallada(resultado: dict, pregunta: dict) -> dict:
    if not pregunta.get("herramienta_esperada"):
        return {"aplica": False, "acierto_trayectoria": None, "errores": []}
    calls = resultado.get("tool_calls_detallado") or []
    errors: list[str] = []
    if not calls:
        errors.append("sin_llamadas")
        return {"aplica": True, "acierto_trayectoria": False,
                "errores": errors}
    unknown = [call.get("name") for call in calls if call.get("name") not in ALLOWED]
    if unknown:
        errors.append("tools_desconocidas")

    family = pregunta.get("familia")
    ticker = pregunta.get("ticker")
    concept = pregunta.get("concept_esperado") or pregunta.get("concept_xbrl")
    names = [call.get("name") for call in calls]
    if family in {"numerica", "comparativa"}:
        incompatible = [name for name in names
                        if name in {"search_filings", "read_section"}]
        if incompatible:
            errors.append("tools_narrativas_innecesarias")
        years = ([int(pregunta["fiscal_year"])] if family == "numerica" else
                 [int(pregunta["fiscal_year_inicio"]),
                  int(pregunta["fiscal_year_fin"])])
        for year in years:
            if not any(_xbrl_matches(call, ticker, year, concept)
                       for call in calls):
                errors.append(f"xbrl_incorrecto_{year}")
        for call in calls:
            if call.get("name") == "get_xbrl_fact" and not any(
                    _xbrl_matches(call, ticker, year, concept) for year in years):
                errors.append("xbrl_incompatible")
    elif family == "extractiva":
        if "get_xbrl_fact" in names:
            errors.append("xbrl_innecesario")
        valid_search_indices = [index for index, call in enumerate(calls)
                                if _search_matches(call, pregunta)]
        if not valid_search_indices:
            errors.append("busqueda_correcta_ausente")
        for index, call in enumerate(calls):
            name = call.get("name")
            if name == "search_filings" and not _search_matches(call, pregunta):
                errors.append("busqueda_incompatible")
            if name == "read_section":
                args = call.get("args") or {}
                correct = (
                    _ticker(args.get("ticker")) == _ticker(ticker)
                    and _year(args.get("fiscal_year")) == int(pregunta["fiscal_year"])
                    and str(args.get("item")) == str(
                        pregunta.get("item", pregunta.get("item_esperado")))
                    and any(search_index < index for search_index
                            in valid_search_indices)
                )
                if not correct:
                    errors.append("read_section_no_es_ultimo_recurso")
    else:
        errors.append("familia_desconocida")

    return {"aplica": True, "acierto_trayectoria": not errors,
            "errores": sorted(set(errors))}


def evaluar_trayectoria(resultado: dict, pregunta: dict) -> bool | None:
    return evaluar_trayectoria_detallada(resultado, pregunta)[
        "acierto_trayectoria"]
