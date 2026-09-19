import sys
from pathlib import Path

# Añadir marco/ al path para que `tools` sea importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import list_available, get_xbrl_fact, search_filings, read_section

print("=== list_available ===")
print(list_available.invoke({}))

print("\n=== get_xbrl_fact ===")
print(get_xbrl_fact.invoke({
    "ticker": "NVDA", "fiscal_year": 2024, "concept": "Revenues"
}))

print("\n=== search_filings ===")
resultado = search_filings.invoke({
    "query": "AI risks", "ticker": "MSFT", "fiscal_year": 2025, "k": 2
})
print(resultado[:400])

print("\n=== read_section ===")
texto = read_section.invoke({"ticker": "AAPL", "fiscal_year": 2024, "item": "7A"})
print(f"Longitud: {len(texto)} caracteres")