"""Resumen de aciertos del baseline."""
import json
from collections import defaultdict
from pathlib import Path

raiz = Path(__file__).resolve().parents[2]
ruta = raiz / "marco" / "results" / "baseline" / "resultados_baseline_oficial.jsonl"

with open(ruta, encoding="utf-8") as f:
    rows = [json.loads(l) for l in f if l.strip()]

print(f"Total: {len(rows)}\n")

for eval_name in ["acierto_cifra", "acierto_cita", "acierto_trayectoria"]:
    aplicables = [r for r in rows if r[eval_name] is not None]
    aciertos = sum(1 for r in aplicables if r[eval_name])
    total = len(aplicables)
    pct = aciertos / total * 100 if total else 0
    print(f"{eval_name}: {aciertos}/{total} = {pct:.1f}%")

print("\nPor familia:")
por_familia = defaultdict(lambda: defaultdict(list))
for r in rows:
    for eval_name in ["acierto_cifra", "acierto_cita", "acierto_trayectoria"]:
        if r[eval_name] is not None:
            por_familia[r["familia"]][eval_name].append(r[eval_name])

for fam, evals in sorted(por_familia.items()):
    print(f"  {fam}:")
    for eval_name, vals in evals.items():
        a = sum(vals)
        print(f"    {eval_name}: {a}/{len(vals)}")

latencias = [r["latencia_s"] for r in rows if r.get("latencia_s")]
if latencias:
    print(f"\nLatencia media: {sum(latencias)/len(latencias):.2f}s")
    print(f"Latencia max: {max(latencias):.2f}s")

errores = [r for r in rows if r.get("error")]
print(f"Errores: {len(errores)}")
for r in errores:
    print(f"  [{r['id']}] {r.get('error')}")