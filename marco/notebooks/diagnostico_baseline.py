"""Diagnóstico de los fallos del baseline. Sin llamadas al LLM."""
import json
from pathlib import Path

raiz = Path(__file__).resolve().parents[2]
ruta = raiz / "marco" / "results" / "baseline" / "resultados_baseline_oficial.jsonl"

with open(ruta, encoding="utf-8") as f:
    rows = [json.loads(l) for l in f if l.strip()]

print("=" * 70)
print("COMPARATIVAS que fallan en cifra (0/7)")
print("=" * 70)
for r in rows:
    if r["familia"] == "comparativa" and r["acierto_cifra"] is False:
        print(f"\n[{r['id']}] {r['pregunta'][:90]}")
        print(f"  cifra_esperada: {r.get('cifra_esperada')}")
        print(f"  cifra_agente:   {r.get('cifra_agente')}")
        print(f"  unidad_agente:  {r.get('unidad_agente')}")
        print(f"  fuente_agente:  {r.get('fuente_agente')}")
        print(f"  respuesta:      {r.get('respuesta_agente', '')[:200]}")

print("\n" + "=" * 70)
print("EXTRACTIVAS que fallan en cita (4/6)")
print("=" * 70)
for r in rows:
    if r["familia"] == "extractiva" and r["acierto_cita"] is False:
        print(f"\n[{r['id']}] {r['pregunta'][:90]}")
        print(f"  chunk_id_agente: {r.get('chunk_id_agente')}")
        print(f"  cita_agente: {r.get('cita_agente', '')[:200] if r.get('cita_agente') else None}")
        print(f"  tool_calls: {r.get('tool_calls_agente')}")