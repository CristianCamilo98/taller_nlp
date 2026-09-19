"""Carga y utilidades XBRL compartidas, perezosas y sin red."""

from __future__ import annotations

from functools import lru_cache

from .config import get_dataset_paths


@lru_cache(maxsize=1)
def load_xbrl():
    import pandas as pd

    return pd.read_parquet(get_dataset_paths().xbrl_facts)


def format_xbrl_value(value: float, unit: str) -> str:
    """Conserva los EPS decimales y representa exactamente los enteros.

    El corpus actual contiene enteros para USD y decimales con dos posiciones
    para USD/shares. ``.15g`` evita inventar decimales a la vez que no redondea
    valores no enteros futuros a un entero.
    """
    number = float(value)
    if number.is_integer():
        return f"{number:,.0f}"
    return f"{number:,.15g}"


def find_xbrl_fact(ticker: str, fiscal_year: int, concept: str):
    facts = load_xbrl()
    return facts[
        (facts["ticker"].astype(str).str.upper() == ticker.strip().upper())
        & (facts["fiscal_year"].astype(int) == int(fiscal_year))
        & (facts["concept"] == concept)
    ]
