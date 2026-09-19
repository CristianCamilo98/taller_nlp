"""Genera el golden_set_grupo3.jsonl completo desde los datos verificados."""
import json
from pathlib import Path
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
xbrl = pd.read_parquet(raiz / "corpus" / "xbrl_facts.parquet")

# Todas las preguntas del golden set propio
PREGUNTAS = [
    # === NUMÉRICAS ===
    {"id": "g3-001", "familia": "numerica", "ticker": "NVDA", "fiscal_year": 2024,
     "concept_xbrl": "Assets",
     "pregunta": "¿Cuál fue el total de activos (Assets) de NVIDIA en el ejercicio fiscal 2024?"},
    {"id": "g3-002", "familia": "numerica", "ticker": "MSFT", "fiscal_year": 2025,
     "concept_xbrl": "NetIncomeLoss",
     "pregunta": "¿Cuál fue el resultado neto (NetIncomeLoss) de Microsoft en el ejercicio fiscal 2025?"},
    {"id": "g3-003", "familia": "numerica", "ticker": "AAPL", "fiscal_year": 2024,
     "concept_xbrl": "EarningsPerShareDiluted",
     "pregunta": "¿Cuál fue el beneficio por acción diluido (EarningsPerShareDiluted) de Apple en el ejercicio fiscal 2024?"},
    {"id": "g3-004", "familia": "numerica", "ticker": "META", "fiscal_year": 2025,
     "concept_xbrl": "CashAndCashEquivalentsAtCarryingValue",
     "pregunta": "¿Cuánto efectivo y equivalentes de efectivo tenía Meta en el ejercicio fiscal 2025?"},
    {"id": "g3-005", "familia": "numerica", "ticker": "GOOGL", "fiscal_year": 2024,
     "concept_xbrl": "OperatingIncomeLoss",
     "pregunta": "¿Cuál fue el resultado operativo (OperatingIncomeLoss) de Alphabet en el ejercicio fiscal 2024?"},
    {"id": "g3-006", "familia": "numerica", "ticker": "AMZN", "fiscal_year": 2025,
     "concept_xbrl": "NetCashProvidedByUsedInOperatingActivities",
     "pregunta": "¿Cuál fue el flujo de caja de actividades operativas de Amazon en el ejercicio fiscal 2025?"},
    {"id": "g3-007", "familia": "numerica", "ticker": "AMZN", "fiscal_year": 2024,
     "concept_xbrl": "NetIncomeLoss",
     "pregunta": "¿Cuál fue el beneficio neto de Amazon en el ejercicio fiscal 2024?"},

    # === EXTRACTIVAS ===
    {"id": "g3-008", "familia": "extractiva", "ticker": "NVDA", "fiscal_year": 2025,
     "item_esperado": "1A",
     "pregunta": "¿Qué riesgo relacionado con los controles a la exportación menciona NVIDIA en sus factores de riesgo de FY2025?",
     "ancla_texto": "As a result, excessive or shifting export controls may negatively impact demand for our products and services not only in China, but also in other markets, such as Europe, Latin America, and Southeast Asia.",
     "ancla_inicio": 85654, "ancla_fin": 85860,
     "chunk_id_esperado": "NVDA-2025-1A-0038",
     "respuesta_esperada": "Que los controles a la exportación pueden impactar negativamente la demanda en China y otros mercados."},
    {"id": "g3-009", "familia": "extractiva", "ticker": "MSFT", "fiscal_year": 2024,
     "item_esperado": "7A",
     "pregunta": "¿Qué dice Microsoft sobre su exposición al riesgo de tipo de cambio en el apartado de riesgo de mercado de FY2024?",
     "ancla_texto": "Certain forecasted transactions, assets, and liabilities are exposed to foreign currency risk.",
     "ancla_inicio": 330, "ancla_fin": 424,
     "chunk_id_esperado": "MSFT-2024-7A-0000",
     "respuesta_esperada": "Que ciertas transacciones, activos y pasivos están expuestos al riesgo de tipo de cambio."},
    {"id": "g3-010", "familia": "extractiva", "ticker": "AAPL", "fiscal_year": 2025,
     "item_esperado": "7",
     "pregunta": "¿Qué explica Apple sobre la evolución de sus ingresos por servicios en la discusión de la dirección de FY2025?",
     "ancla_texto": "Services net sales increased during 2025 compared to 2024 primarily due to higher net sales from advertising, the App Store and cloud services.",
     "ancla_inicio": 6437, "ancla_fin": 6580,
     "chunk_id_esperado": "AAPL-2025-7-0003",
     "respuesta_esperada": "Que aumentaron principalmente por mayores ventas de publicidad, App Store y servicios en la nube."},
    {"id": "g3-011", "familia": "extractiva", "ticker": "META", "fiscal_year": 2024,
     "item_esperado": "1A",
     "pregunta": "¿Qué riesgo regulatorio relacionado con privacidad de datos menciona Meta en sus factores de riesgo de FY2024?",
     "ancla_texto": "the interpretation and enforcement of the GDPR, as well as the imposition and amount of penalties for non-compliance, are subject to significant uncertainty",
     "ancla_inicio": 130644, "ancla_fin": 130800,
     "chunk_id_esperado": "META-2024-1A-0057",
     "respuesta_esperada": "Que la interpretación y aplicación del GDPR, así como las penalizaciones, están sujetas a incertidumbre significativa."},
    {"id": "g3-012", "familia": "extractiva", "ticker": "GOOGL", "fiscal_year": 2025,
     "item_esperado": "7A",
     "pregunta": "¿Qué dice Alphabet sobre su exposición al riesgo de divisa extranjera en el apartado de riesgo de mercado de FY2025?",
     "ancla_texto": "International revenues, foreign-denominated monetary assets and liabilities, and investments in foreign subsidiaries expose us to the risk of fluctuations in foreign exchange rates against the US dollar.",
     "ancla_inicio": 296, "ancla_fin": 499,
     "chunk_id_esperado": "GOOGL-2025-7A-0000",
     "respuesta_esperada": "Que sus ingresos internacionales, activos y pasivos monetarios en divisas e inversiones en subsidiarias extranjeras la exponen al riesgo de fluctuaciones del tipo de cambio."},
    {"id": "g3-013", "familia": "extractiva", "ticker": "AMZN", "fiscal_year": 2024,
     "item_esperado": "8",
     "pregunta": "¿Qué políticas contables relevantes describe Amazon en sus estados financieros de FY2024?",
     "ancla_texto": "We utilize a two-step approach to recognizing and measuring uncertain income tax positions (income tax contingencies).",
     "ancla_inicio": 36440, "ancla_fin": 36558,
     "chunk_id_esperado": "AMZN-2024-8-0023",
     "respuesta_esperada": "Que utiliza un enfoque de dos pasos para reconocer y medir posiciones fiscales inciertas."},

    # === COMPARATIVAS ===
    {"id": "g3-014", "familia": "comparativa", "ticker": "NVDA", "fiscal_year": 2025,
     "concept_xbrl": "Revenues",
     "pregunta": "¿Cuánto creció el revenue de NVIDIA entre FY2024 y FY2025?"},
    {"id": "g3-015", "familia": "comparativa", "ticker": "AAPL", "fiscal_year": 2025,
     "concept_xbrl": "NetIncomeLoss",
     "pregunta": "¿Cuánto creció el beneficio neto de Apple entre FY2024 y FY2025?"},
    {"id": "g3-016", "familia": "comparativa", "ticker": "META", "fiscal_year": 2025,
     "concept_xbrl": "ResearchAndDevelopmentExpense",
     "pregunta": "¿Cuánto creció el gasto en I+D de Meta entre FY2024 y FY2025?"},
    {"id": "g3-017", "familia": "comparativa", "ticker": "GOOGL", "fiscal_year": 2025,
     "concept_xbrl": "Revenues",
     "pregunta": "¿Cuánto creció el revenue de Alphabet entre FY2024 y FY2025?"},
    {"id": "g3-018", "familia": "comparativa", "ticker": "AAPL", "fiscal_year": 2025,
     "concept_xbrl": "EarningsPerShareDiluted",
     "pregunta": "¿Cuánto creció el beneficio por acción diluido de Apple entre FY2024 y FY2025?"},
    {"id": "g3-019", "familia": "comparativa", "ticker": "NVDA", "fiscal_year": 2025,
        "concept_xbrl": "OperatingIncomeLoss",
        "pregunta": "¿Cuánto creció el beneficio operativo (OperatingIncomeLoss) de NVIDIA entre FY2024 y FY2025?"},
    {"id": "g3-020", "familia": "comparativa", "ticker": "GOOGL", "fiscal_year": 2025,
     "concept_xbrl": "StockholdersEquity",
     "pregunta": "¿Cuánto creció el patrimonio neto de Alphabet entre FY2024 y FY2025?"},
]


def rellenar_numerica(p):
    f = xbrl[(xbrl.ticker == p["ticker"]) & (xbrl.fiscal_year == p["fiscal_year"]) & (xbrl.concept == p["concept_xbrl"])].iloc[0]
    p["cifra_esperada"] = float(f.value)
    p["unidad"] = f.unit
    p["respuesta_esperada"] = f"{f.value:,.0f} {f.unit}"
    return p


def rellenar_comparativa(p):
    v_act = xbrl[(xbrl.ticker == p["ticker"]) & (xbrl.fiscal_year == p["fiscal_year"]) & (xbrl.concept == p["concept_xbrl"])].iloc[0]
    v_ant = xbrl[(xbrl.ticker == p["ticker"]) & (xbrl.fiscal_year == p["fiscal_year"] - 1) & (xbrl.concept == p["concept_xbrl"])].iloc[0]
    delta = float(v_act.value) - float(v_ant.value)
    pct = delta / abs(float(v_ant.value)) * 100
    p["cifra_esperada"] = float(v_act.value)
    p["unidad"] = v_act.unit
    p["respuesta_esperada"] = f"Creció de {v_ant.value:,.0f} a {v_act.value:,.0f} {v_act.unit} (delta {delta:+,.0f}, {pct:+.1f}%)"
    return p


def normalizar(p):
    """Asegura que todos los campos del esquema están presentes."""
    campos = ["respuesta_esperada", "cifra_esperada", "unidad", "concept_xbrl",
              "item_esperado", "ancla_texto", "ancla_inicio", "ancla_fin",
              "chunk_id_esperado", "herramienta_esperada"]
    for c in campos:
        p.setdefault(c, None)
    # Herramienta esperada según familia
    if p["familia"] == "numerica":
        p["herramienta_esperada"] = ["get_xbrl_fact"]
    elif p["familia"] == "extractiva":
        p["herramienta_esperada"] = ["search_filings"]
    elif p["familia"] == "comparativa":
        if p.get("concept_xbrl"):
            p["herramienta_esperada"] = ["get_xbrl_fact", "search_filings"]
        else:
            p["herramienta_esperada"] = ["search_filings"]
    p["autor"] = "marco"
    return p


def main():
    completadas = []
    for p in PREGUNTAS:
        p = normalizar(p)
        try:
            if p["familia"] == "numerica":
                p = rellenar_numerica(p)
            elif p["familia"] == "comparativa" and p.get("concept_xbrl"):
                p = rellenar_comparativa(p)
            completadas.append(p)
        except Exception as e:
            print(f"❌ {p['id']}: {e}")

    salida = raiz / "marco" / "golden_set" / "golden_set_grupo3.jsonl"
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8", newline="\n") as f:
        for p in completadas:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n✅ {len(completadas)} preguntas guardadas en {salida}")
    print(f"   Faltan: {20 - len(completadas)} (g3-019 pendiente)")


if __name__ == "__main__":
    main()