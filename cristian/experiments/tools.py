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
    """Lista las compañías, ejercicios y secciones disponibles en el corpus.

    Úsala antes de afirmar que un dato no existe y cuando no conozcas la
    cobertura del corpus.
    """
    sections = _load_sections()
    lines = ["El corpus contiene los siguientes datos:"]
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
    return "\n".join(lines)


@tool
def get_xbrl_fact(ticker: str, fiscal_year: int, concept: str) -> str:
    """Devuelve un hecho financiero exacto del XBRL autorizado.

    Args:
        ticker: Símbolo bursátil, por ejemplo ``NVDA``.
        fiscal_year: Ejercicio fiscal reportado, por ejemplo ``2024``.
        concept: Concepto US-GAAP exacto, por ejemplo ``Revenues``.

    Es la fuente obligatoria para cifras. Nunca deriva ni estima valores
    ausentes.
    """
    ticker = ticker.strip().upper()
    year = int(fiscal_year)
    facts = load_xbrl()
    rows = facts[
        (facts["ticker"].astype(str).str.upper() == ticker)
        & (facts["fiscal_year"].astype(int) == year)
        & (facts["concept"] == concept)
    ]
    if rows.empty:
        available = sorted(
            facts[
                (facts["ticker"].astype(str).str.upper() == ticker)
                & (facts["fiscal_year"].astype(int) == year)
            ]["concept"].unique()
        )
        if not available:
            return (
                f"No hay datos de {ticker} para FY{year} en el corpus. "
                "Usa list_available para consultar la cobertura."
            )
        return (
            f"{ticker} no reportó '{concept}' en FY{year}. Conceptos "
            f"disponibles: {', '.join(available)}"
        )
    row = rows.iloc[0]
    value = format_xbrl_value(float(row["value"]), str(row["unit"]))
    return (
        f"{ticker} FY{year} · {concept} = {value} {row['unit']} "
        f"(cierre de ejercicio {row['period_end']}, según el {row['form']})"
    )


@tool
def search_filings(
    query: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    k: int = 5,
) -> str:
    """Busca fragmentos relevantes de los informes 10-K.

    Args:
        query: Consulta semántica en inglés.
        ticker: Filtro opcional por compañía.
        fiscal_year: Filtro opcional por ejercicio.
        item: Filtro opcional: ``1A``, ``7``, ``7A`` u ``8``.
        k: Número positivo de fragmentos solicitados; por defecto 5.

    Úsala para evidencia narrativa, no para cifras.
    """
    if int(k) <= 0:
        raise ValueError("k debe ser un entero positivo")
    from cristian.experiments.miax_s1 import buscar, formatear_fragmentos

    fragments = buscar(
        query,
        ticker=ticker.strip().upper() if ticker else None,
        fiscal_year=int(fiscal_year) if fiscal_year is not None else None,
        item=str(item) if item is not None else None,
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
