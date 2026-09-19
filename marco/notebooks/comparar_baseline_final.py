"""Compara baseline vs final sobre el mismo golden set."""
import json
from collections import defaultdict
from pathlib import Path

raiz = Path(__file__).resolve().parents[2]


def cargar(ruta):
    return [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]


def resumen(rows):
    out = {}
    for e in ["acierto_cifra", "acierto_cita", "acierto_trayectoria"]:
        ap = [r for r in rows if r[e] is not None]
        ac = sum(1 for r in ap if r[e])
        out[e] = (ac, len(ap))
    lat = [r["latencia_s"] for r in rows if r.get("latencia_s")]
    out["latencia_media"] = sum(lat) / len(lat) if lat else 0
    return out


def por_familia(rows):
    pf = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for e in ["acierto_cifra", "acierto_cita", "acierto_trayectoria"]:
            if r[e] is not None:
                pf[r["familia"]][e].append(r[e])
    return pf


baseline = cargar(raiz / "marco/results/baseline/BASELINE_OFICIAL.jsonl")
final = cargar(raiz / "marco/results/final/resultados_final_v1.jsonl")

rb, rf = resumen(baseline), resumen(final)
print("=" * 60)
print("COMPARATIVA BASELINE vs FINAL v1 (query rewriting)")
print("=" * 60)
for e in ["acierto_cifra", "acierto_cita", "acierto_trayectoria"]:
    ab, tb = rb[e]
    af, tf = rf[e]
    pb = ab / tb * 100 if tb else 0
    pf = af / tf * 100 if tf else 0
    delta = pf - pb
    signo = "+" if delta >= 0 else ""
    print(f"{e}:")
    print(f"  baseline: {ab}/{tb} = {pb:.1f}%")
    print(f"  final:    {af}/{tf} = {pf:.1f}%  ({signo}{delta:.1f}pp)")
print(f"latencia media:")
print(f"  baseline: {rb['latencia_media']:.2f}s")
print(f"  final:    {rf['latencia_media']:.2f}s")

print("\n" + "=" * 60)
print("POR FAMILIA (final)")
print("=" * 60)
pf_fam = por_familia(final)
for f, evs in sorted(pf_fam.items()):
    print(f"{f}:")
    for e, v in evs.items():
        print(f"  {e}: {sum(v)}/{len(v)}")

# Detectar preguntas que cambiaron
print("\n" + "=" * 60)
print("PREGUNTAS QUE CAMBIARON")
print("=" * 60)
b_por_id = {r["id"]: r for r in baseline}
f_por_id = {r["id"]: r for r in final}
for pid in sorted(b_por_id.keys()):
    for e in ["acierto_cifra", "acierto_cita", "acierto_trayectoria"]:
        b_val = b_por_id[pid][e]
        f_val = f_por_id[pid][e]
        if b_val != f_val:
            print(f"[{pid}] {e}: {b_val} → {f_val}")