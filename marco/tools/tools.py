"""Las 4 herramientas del agente sobre informes 10-K.

Contrato del enunciado (NO cambiar nombres ni parámetros):
    list_available() -> str
    get_xbrl_fact(ticker, fiscal_year, concept) -> str
    search_filings(query, ticker=None, fiscal_year=None, item=None, k=5) -> str
    read_section(ticker, fiscal_year, item) -> str
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from langchain.tools import tool


# ---------------------------------------------------------------------------
# Localizar corpus/ y miax_s1.py sin depender del CWD
# ---------------------------------------------------------------------------
def _encontrar_corpus() -> Path:
    candidatos = [
        Path("corpus"),
        Path(__file__).resolve().parent.parent.parent / "corpus",
        Path(__file__).resolve().parents[3] / "corpus",
    ]
    for c in candidatos:
        if (c / "chunks.jsonl").is_file():
            return c
    raise RuntimeError(
        f"No encuentro corpus/chunks.jsonl. Buscado en: "
        f"{[str(c) for c in candidatos]}. Ejecuta desde la raíz del repo."
    )


def _encontrar_miax_s1() -> Path:
    candidatos = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent / "starter",
        Path(__file__).resolve().parents[2] / "Clase_1" / "Clase_1",
        Path(__file__).resolve().parents[3] / "Clase_1" / "Clase_1",
    ]
    for c in candidatos:
        if (c / "miax_s1.py").is_file():
            return c
    raise RuntimeError(
        f"No encuentro miax_s1.py. Buscado en: "
        f"{[str(c) for c in candidatos]}"
    )


CORPUS = _encontrar_corpus()
_RUTA_MIAX = _encontrar_miax_s1()
if str(_RUTA_MIAX) not in sys.path:
    sys.path.insert(0, str(_RUTA_MIAX))

import miax_s1  # noqa: E402


# ---------------------------------------------------------------------------
# Carga única de datos
# ---------------------------------------------------------------------------
secciones = pd.DataFrame(
    json.loads(l) for l in open(CORPUS / "secciones.jsonl", encoding="utf-8")
)
xbrl = pd.read_parquet(CORPUS / "xbrl_facts.parquet")


# ---------------------------------------------------------------------------
# 1. list_available
# ---------------------------------------------------------------------------
@tool
def list_available() -> str:
    """Lista qué compañías, ejercicios y secciones existen en el corpus.

    Úsala SIEMPRE antes de responder que un dato no existe, y antes de
    llamar a cualquier otra herramienta si no estás seguro de que la
    compañía o el ejercicio que te piden estén en el corpus.
    """
    lineas = ["El corpus contiene los siguientes datos:\n"]
    for ticker in sorted(secciones.ticker.unique()):
        sub = secciones[secciones.ticker == ticker]
        empresa = sub.empresa.iloc[0]
        años = sorted(int(a) for a in sub.fiscal_year.unique())
        items = sorted(sub.item.unique())
        años_str = ", ".join(str(a) for a in años)
        items_str = ", ".join(items)
        lineas.append(
            f"- {ticker} ({empresa}): ejercicios {años_str}, items {items_str}"
        )
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# 2. get_xbrl_fact
# ---------------------------------------------------------------------------
@tool
def get_xbrl_fact(ticker: str, fiscal_year: int, concept: str) -> str:
    """Devuelve el valor EXACTO de una magnitud financiera tal y como la
    compañía la reportó en XBRL.

    Es la fuente autorizada para cualquier cifra. Úsala SIEMPRE en lugar de
    leer un número del texto del informe.

    Args:
        ticker: Símbolo bursátil, p. ej. 'NVDA'.
        fiscal_year: Ejercicio fiscal reportado, p. ej. 2024.
        concept: Concepto en taxonomía US-GAAP, p. ej. 'Revenues',
            'NetIncomeLoss', 'Assets', 'OperatingIncomeLoss'.

    Devuelve el valor con su unidad y fecha de cierre, o un aviso explícito
    si la compañía no reportó ese concepto en ese ejercicio.
    """
    filas = xbrl[(xbrl.ticker == ticker)
                 & (xbrl.fiscal_year == int(fiscal_year))
                 & (xbrl.concept == concept)]
    if filas.empty:
        disponibles = sorted(
            xbrl[(xbrl.ticker == ticker)
                 & (xbrl.fiscal_year == int(fiscal_year))].concept.unique()
        )
        if not disponibles:
            return (f"No hay datos de {ticker} para FY{fiscal_year} en el "
                    f"corpus. Usa list_available para ver qué hay.")
        return (f"{ticker} no reportó '{concept}' en FY{fiscal_year}. "
                f"Conceptos disponibles: {', '.join(disponibles)}")
    f = filas.iloc[0]
    return (f"{ticker} FY{fiscal_year} · {concept} = {f.value:,.0f} {f.unit} "
            f"(cierre de ejercicio {f.period_end}, según el {f.form})")


# ---------------------------------------------------------------------------
# 3. search_filings
# ---------------------------------------------------------------------------
@tool
def search_filings(query: str, ticker: str | None = None,
                   fiscal_year: int | None = None,
                   item: str | None = None, k: int = 5) -> str:
    """Busca fragmentos de texto relevantes en los informes 10-K del corpus.

    Úsala para preguntas cualitativas: riesgos, estrategia, litigios,
    comentarios de la dirección. NO la uses para obtener cifras: para eso
    está get_xbrl_fact.

    Args:
        query: Qué buscar, en lenguaje natural. El corpus está en inglés,
            así que escribe la consulta en inglés.
        ticker: Filtra por compañía si la pregunta la menciona.
        fiscal_year: Filtra por ejercicio si la pregunta lo menciona.
        item: Filtra por sección: '1A' riesgos, '7' MD&A,
            '7A' riesgo de mercado, '8' estados financieros.
        k: Número de fragmentos a devolver.

    Devuelve k fragmentos, cada uno con su chunk_id para poder citarlo.
    """
    return miax_s1.formatear_fragmentos(
        miax_s1.buscar(query, ticker=ticker, fiscal_year=fiscal_year,
                       item=item, k=k)
    )


# ---------------------------------------------------------------------------
# 4. read_section
# ---------------------------------------------------------------------------
@tool
def read_section(ticker: str, fiscal_year: int, item: str) -> str:
    """Devuelve el TEXTO COMPLETO de una sección de un 10-K.

    Es una herramienta CARA: puede devolver decenas de miles de tokens.
    Úsala solo cuando search_filings devuelva fragmentos insuficientes y
    necesites el contexto entero de una sección concreta.

    Args:
        ticker: Símbolo bursátil, p. ej. 'META'.
        fiscal_year: Ejercicio fiscal, p. ej. 2025.
        item: '1A' riesgos, '7' MD&A, '7A' riesgo de mercado,
            '8' estados financieros.
    """
    filas = secciones[(secciones.ticker == ticker)
                      & (secciones.fiscal_year == int(fiscal_year))
                      & (secciones.item == item)]
    if filas.empty:
        return (f"No hay Item {item} de {ticker} FY{fiscal_year} en el "
                f"corpus. Usa list_available para ver qué hay.")
    return filas.iloc[0].texto