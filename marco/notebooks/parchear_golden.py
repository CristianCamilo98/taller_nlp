"""Parchea herramienta_esperada en el golden set según el contenido real."""
import json
from pathlib import Path

raiz = Path(__file__).resolve().parents[2]
ruta = raiz / "marco/golden_set/golden_set_grupo3.jsonl"

preguntas = [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]

for p in preguntas:
    if p["familia"] == "comparativa":
        tiene_concept = bool(p.get("concept_xbrl"))
        tiene_ancla = bool(p.get("ancla_texto"))
        if tiene_concept and tiene_ancla:
            p["herramienta_esperada"] = ["get_xbrl_fact", "search_filings"]
        elif tiene_concept:
            # Comparativa puramente numérica: solo XBRL
            p["herramienta_esperada"] = ["get_xbrl_fact"]
        else:
            p["herramienta_esperada"] = ["search_filings"]
    # (dejar numericas y extractivas como están)

with open(ruta, "w", encoding="utf-8", newline="\n") as f:
    for p in preguntas:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

# Resumen
print("Resumen de herramienta_esperada por familia:")
for p in preguntas:
    print(f"  [{p['id']}] {p['familia']:12s} → {p['herramienta_esperada']}")