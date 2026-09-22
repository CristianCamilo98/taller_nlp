"""Las cuatro herramientas del agente experimental.

Misma firma que el enunciado (contrato). Implementación propia: datos y
retrieval viven en este paquete (`config`, `xbrl`, `miax_s1`), no en common.
"""

from __future__ import annotations

import json
from functools import lru_cache

from langchain.tools import tool

from cristian.experiments.config import get_dataset_paths
from cristian.experiments.xbrl import format_xbrl_value, load_xbrl


@lru_cache(maxsize=1)
def _load_sections():
    import pandas as pd

    with get_dataset_paths().sections.open(encoding="utf-8") as stream:
        return pd.DataFrame(
            json.loads(line) for line in stream if line.strip()
        )


@tool
def list_available() -> str:
    """Lista cobertura del corpus textual y de los hechos XBRL.

    Incluye compañías, ejercicios, secciones 10-K (``1A``, ``7``, ``7A``,
    ``8``) y los conceptos US-GAAP disponibles en ``xbrl_facts`` (globales
    y por ticker: no todas las empresas reportan los mismos conceptos).

    Úsala antes de afirmar que un dato no existe, cuando no conozcas la
    cobertura del corpus, qué ``concept`` pasar a ``get_xbrl_fact``, o qué
    ``item`` pasar a ``search_filings``. Como mucho una vez por pregunta.
    """
    sections = _load_sections()
    facts = load_xbrl()
    lines = ["## Corpus textual (secciones 10-K)"]
    for ticker in sorted(sections["ticker"].unique()):
        subset = sections[sections["ticker"] == ticker]
        company = subset["empresa"].iloc[0]
        years = ", ".join(
            str(int(year))
            for year in sorted(subset["fiscal_year"].unique())
        )
        items = ", ".join(sorted(subset["item"].unique()))
        lines.append(
            f"- {ticker} ({company}): ejercicios {years}, items {items}"
        )

    concepts = sorted(facts["concept"].astype(str).unique())
    lines.append("")
    lines.append("## Conceptos XBRL únicos en el corpus")
    lines.append(", ".join(concepts))

    lines.append("")
    lines.append(
        "## Conceptos XBRL por ticker "
        "(solo los que esa compañía reporta)"
    )
    for ticker in sorted(facts["ticker"].astype(str).str.upper().unique()):
        subset = facts[facts["ticker"].astype(str).str.upper() == ticker]
        years = ", ".join(
            str(int(year))
            for year in sorted(subset["fiscal_year"].unique())
        )
        ticker_concepts = ", ".join(
            sorted(subset["concept"].astype(str).unique())
        )
        lines.append(f"- {ticker} (FY {years}): {ticker_concepts}")

    return "\n".join(lines)


@tool
def get_xbrl_fact(ticker: str, fiscal_year: int, concept: str) -> str:
    """Devuelve un hecho financiero exacto del XBRL autorizado.

    Args:
        ticker: Símbolo bursátil, por ejemplo ``NVDA``.
        fiscal_year: Ejercicio fiscal reportado, por ejemplo ``2024``.
        concept: Concepto US-GAAP exacto, por ejemplo ``Revenues``.
            Debe coincidir con un valor de la columna ``concept`` de
            ``xbrl_facts`` (consulta ``list_available``). No todas las
            compañías usan el mismo concepto de ingresos.

    Es la fuente obligatoria para cifras. Nunca deriva ni estima valores
    ausentes. Si el concepto existe para otras empresas pero no para este
    ticker, lo indica explícitamente.
    """
    ticker = ticker.strip().upper()
    year = int(fiscal_year)
    concept = str(concept).strip()
    facts = load_xbrl()
    tickers = facts["ticker"].astype(str).str.upper()
    years = facts["fiscal_year"].astype(int)
    concepts = facts["concept"].astype(str)

    rows = facts[(tickers == ticker) & (years == year) & (concepts == concept)]
    if not rows.empty:
        row = rows.iloc[0]
        value = format_xbrl_value(float(row["value"]), str(row["unit"]))
        return (
            f"{ticker} FY{year} · {concept} = {value} {row['unit']} "
            f"(cierre de ejercicio {row['period_end']}, según el {row['form']})"
        )

    # Fallos: distinguir "no hay ese ticker/año" vs "concepto de otros".
    ticker_year = facts[(tickers == ticker) & (years == year)]
    if ticker_year.empty:
        ticker_years = sorted(
            int(y) for y in facts.loc[tickers == ticker, "fiscal_year"].unique()
        )
        if not ticker_years:
            return (
                f"No hay datos XBRL de {ticker} en el corpus. "
                "Usa list_available para consultar la cobertura."
            )
        return (
            f"No hay datos de {ticker} para FY{year} en el corpus. "
            f"Ejercicios disponibles para {ticker}: "
            f"{', '.join(f'FY{y}' for y in ticker_years)}. "
            "Usa list_available para consultar la cobertura."
        )

    available_here = sorted(ticker_year["concept"].astype(str).unique())
    concept_elsewhere = facts[concepts == concept]
    if not concept_elsewhere.empty:
        other_tickers = sorted(
            concept_elsewhere["ticker"].astype(str).str.upper().unique()
        )
        other_tickers = [t for t in other_tickers if t != ticker]
        # ¿El mismo ticker lo tiene en otro ejercicio?
        same_ticker_other_years = sorted(
            int(y)
            for y in concept_elsewhere.loc[
                concept_elsewhere["ticker"].astype(str).str.upper() == ticker,
                "fiscal_year",
            ].unique()
            if int(y) != year
        )
        if other_tickers and ticker not in set(
            concept_elsewhere["ticker"].astype(str).str.upper().unique()
        ):
            return (
                f"ERROR: el concepto '{concept}' existe en el corpus XBRL "
                f"para otras empresas ({', '.join(other_tickers)}), pero "
                f"NO está reportado para {ticker}. "
                f"No inventes ni sustituyas el valor. "
                f"Conceptos disponibles para {ticker} en FY{year}: "
                f"{', '.join(available_here)}."
            )
        if same_ticker_other_years:
            return (
                f"{ticker} no reportó '{concept}' en FY{year}, pero sí en "
                f"{', '.join(f'FY{y}' for y in same_ticker_other_years)}. "
                f"Conceptos disponibles para {ticker} en FY{year}: "
                f"{', '.join(available_here)}."
            )
        # Concepto en el corpus, mismo ticker en este año no (ya cubierto),
        # o solo otros tickers con solapamiento raro — mensaje genérico útil.
        where = ", ".join(
            f"{r.ticker} FY{int(r.fiscal_year)}"
            for r in concept_elsewhere[
                ["ticker", "fiscal_year"]
            ].drop_duplicates().itertuples(index=False)
        )
        return (
            f"ERROR: '{concept}' no está disponible para {ticker} FY{year}. "
            f"En el corpus aparece en: {where}. "
            f"Conceptos disponibles para {ticker} en FY{year}: "
            f"{', '.join(available_here)}."
        )

    all_concepts = sorted(concepts.unique())
    return (
        f"{ticker} no reportó '{concept}' en FY{year} y ese concepto "
        f"tampoco aparece en el corpus XBRL. "
        f"Conceptos disponibles para {ticker} en FY{year}: "
        f"{', '.join(available_here)}. "
        f"Conceptos globales del corpus: {', '.join(all_concepts)}."
    )


@tool
def search_filings(
    query: str,
    ticker: str,
    fiscal_year: int,
    item: str,
    k: int = 5,
) -> str:
    """Busca fragmentos de un 10-K filtrados por compañía, ejercicio y sección.

    ``item`` es obligatorio. No llames a esta herramienta sin uno de estos
    códigos: ``1A`` factores de riesgo, ``7`` MD&A, ``7A`` riesgo de mercado,
    ``8`` estados financieros y notas. Si la pregunta no permite elegir el
    código, llama antes a ``list_available`` (una sola vez) y reintenta con
    el item que devuelva. No repitas esta llamada sin item.

    Args:
        query: Consulta semántica en inglés.
        ticker: Símbolo bursátil, por ejemplo ``META``.
        fiscal_year: Ejercicio fiscal, por ejemplo ``2024``.
        item: Sección obligatoria: ``1A``, ``7``, ``7A`` u ``8``.
        k: Número positivo de fragmentos; por defecto 5.

    Úsala para evidencia narrativa, no para cifras.
    """
    if int(k) <= 0:
        raise ValueError("k debe ser un entero positivo")
    item_code = str(item).strip().upper() if item is not None else ""
    if item_code not in {"1A", "7", "7A", "8"}:
        return (
            "search_filings no se ha ejecutado: falta un item válido "
            "(1A factores de riesgo, 7 MD&A, 7A riesgo de mercado, "
            "8 estados financieros). "
            "Si no puedes deducirlo de la pregunta, llama a list_available "
            "una vez y vuelve a buscar con ese item. "
            "No repitas esta llamada sin item."
        )
    from cristian.experiments.miax_s1 import buscar, formatear_fragmentos

    fragments = buscar(
        query,
        ticker=ticker.strip().upper() if ticker else None,
        fiscal_year=int(fiscal_year) if fiscal_year is not None else None,
        item=item_code,
        k=int(k),
    )
    return formatear_fragmentos(fragments)


@tool
def read_section(ticker: str, fiscal_year: int, item: str) -> str:
    """Devuelve el texto completo de una sección 10-K.

    Args:
        ticker: Símbolo bursátil.
        fiscal_year: Ejercicio fiscal.
        item: Sección ``1A``, ``7``, ``7A`` u ``8``.

    Es una herramienta cara y de último recurso: úsala solo tras una búsqueda
    insuficiente en la misma compañía, ejercicio y sección.
    """
    ticker = ticker.strip().upper()
    year = int(fiscal_year)
    item = str(item)
    sections = _load_sections()
    rows = sections[
        (sections["ticker"].astype(str).str.upper() == ticker)
        & (sections["fiscal_year"].astype(int) == year)
        & (sections["item"].astype(str) == item)
    ]
    if rows.empty:
        return (
            f"No hay Item {item} de {ticker} FY{year} en el corpus. "
            "Usa list_available para consultar la cobertura."
        )
    return str(rows.iloc[0]["texto"])
