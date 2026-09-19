"""Valida cada pregunta del golden set contra el corpus real.

Uso:
    python marco/notebooks/validar_golden.py <ruta_golden.jsonl>
"""
import json
import sys
from pathlib import Path

import pandas as pd

raiz = Path(__file__).resolve().parents[2]
secciones = pd.DataFrame(
    json.loads(l) for l in open(raiz / "corpus" / "secciones.jsonl", encoding="utf-8")
)
xbrl = pd.read_parquet(raiz / "corpus" / "xbrl_facts.parquet")


def validar_pregunta(p: dict) -> list[str]:
    """Devuelve la lista de problemas encontrados (vacía = OK)."""
    errores = []
    pid = p.get("id", "?")
    familia = p.get("familia")

    # --- Comprobaciones básicas ---
    if p.get("ticker") not in set(secciones.ticker):
        errores.append(f"ticker {p.get('ticker')} no está en el corpus")
    if int(p.get("fiscal_year", 0)) not in set(secciones.fiscal_year):
        errores.append(f"fiscal_year {p.get('fiscal_year')} no está en el corpus")
    if familia not in {"numerica", "extractiva", "comparativa"}:
        errores.append(f"familia '{familia}' no válida")

    # --- Verificación XBRL (numéricas y comparativas con concept) ---
    tiene_concept = bool(p.get("concept_xbrl"))
    if familia == "numerica" or (familia == "comparativa" and tiene_concept):
        if not tiene_concept:
            errores.append("falta concept_xbrl")
        else:
            fila = xbrl[
                (xbrl.ticker == p["ticker"])
                & (xbrl.fiscal_year == int(p["fiscal_year"]))
                & (xbrl.concept == p["concept_xbrl"])
            ]
            if fila.empty:
                otros = xbrl[
                    (xbrl.ticker == p["ticker"]) & (xbrl.concept == p["concept_xbrl"])
                ]
                if otros.empty:
                    errores.append(
                        f"{p['ticker']} NO reporta '{p['concept_xbrl']}' en ningún año. "
                        f"Disponibles: {sorted(xbrl[xbrl.ticker == p['ticker']].concept.unique())}"
                    )
                else:
                    años = sorted(int(a) for a in otros.fiscal_year.unique())
                    errores.append(
                        f"{p['ticker']} reporta '{p['concept_xbrl']}' solo en años "
                        f"{años}, no en FY{p['fiscal_year']}"
                    )
            else:
                valor_real = fila.iloc[0].value
                valor_esperado = p.get("cifra_esperada")
                if valor_esperado is None:
                    errores.append("falta cifra_esperada")
                elif abs(valor_real - valor_esperado) / max(abs(valor_real), 1) > 0.005:
                    errores.append(
                        f"cifra_esperada {valor_esperado} NO coincide con el corpus "
                        f"({valor_real})"
                    )

    # --- Verificación del ancla ---
    # Extractivas: SIEMPRE necesitan ancla.
    # Comparativas: solo si NO tienen concept_xbrl (son cualitativas puras).
    necesita_ancla = (
        familia == "extractiva"
        or (familia == "comparativa" and not tiene_concept)
    )

    if necesita_ancla:
        ancla = p.get("ancla_texto")
        if not ancla:
            errores.append(f"{familia} sin ancla_texto")
        else:
            item = p.get("item_esperado")
            candidatas = secciones[
                (secciones.ticker == p["ticker"])
                & (secciones.fiscal_year == int(p["fiscal_year"]))
            ]
            if item:
                candidatas = candidatas[candidatas.item == item]

            encontrada = False
            for _, fila in candidatas.iterrows():
                if ancla in fila.texto:
                    encontrada = True
                    inicio = fila.texto.find(ancla)
                    if p.get("ancla_inicio") is not None:
                        if abs(p["ancla_inicio"] - inicio) > 5:
                            errores.append(
                                f"ancla_inicio declarado ({p['ancla_inicio']}) no coincide "
                                f"con la posición real ({inicio})"
                            )
                    break
            if not encontrada:
                errores.append(
                    f"ancla_texto NO aparece literalmente en {p['ticker']} "
                    f"FY{p['fiscal_year']} Item {item}"
                )
            if len(ancla.split()) > 40:
                errores.append(f"ancla de {len(ancla.split())} palabras (>40)")

    # --- Verificación de campos obligatorios ---
    if not p.get("herramienta_esperada"):
        errores.append("falta herramienta_esperada")

    return errores


def main(ruta: str) -> None:
    preguntas = [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]
    print(f"Validando {len(preguntas)} preguntas de {ruta}\n")

    total_errores = 0
    for p in preguntas:
        errores = validar_pregunta(p)
        if errores:
            print(f"❌ [{p['id']}] {p['pregunta'][:70]}...")
            for e in errores:
                print(f"   - {e}")
            total_errores += len(errores)
        else:
            print(f"✅ [{p['id']}] OK")

    print(f"\nTotal: {total_errores} errores")
    if total_errores == 0:
        print("🎉 Golden set válido")
    else:
        print("⚠️ Corrige los errores antes de congelar")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python validar_golden.py <ruta_golden.jsonl>")
        sys.exit(1)
    main(sys.argv[1])