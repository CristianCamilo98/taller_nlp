"""Re-evalúa el baseline v5 sin llamar al LLM (con golden set corregido)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.evaluador_cifra import evaluar_cifra
from eval.evaluador_cita import evaluar_cita
from eval.evaluador_trayectoria import evaluar_trayectoria

raiz = Path(__file__).resolve().parents[2]
ruta_resultados = raiz / "marco/results/baseline/BASELINE_OFICIAL_v5.jsonl"
ruta_golden = raiz / "marco/golden_set/golden_set_grupo3.jsonl"

golden = {json.loads(l)["id"]: json.loads(l)
          for l in open(ruta_golden, encoding="utf-8") if l.strip()}
rows = [json.loads(l) for l in open(ruta_resultados, encoding="utf-8") if l.strip()]

for r in rows:
    p = golden[r["id"]]
    if r.get("error"):
        r["acierto_cifra"] = False if p["familia"] in {"numerica", "comparativa"} else None
        r["acierto_cita"] = False if p.get("ancla_texto") else None
        r["acierto_trayectoria"] = False
    else:
        r["acierto_cifra"] = evaluar_cifra(r, p)
        r["acierto_cita"] = evaluar_cita(r, p)
        r["acierto_trayectoria"] = evaluar_trayectoria(r, p)

salida = raiz / "marco/results/baseline/BASELINE_OFICIAL_v5_reeval.jsonl"
with open(salida, "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import defaultdict
for e in ["acierto_cifra", "acierto_cita", "acierto_trayectoria"]:
    ap = [r for r in rows if r[e] is not None]
    ac = sum(1 for r in ap if r[e])
    print(f"{e}: {ac}/{len(ap)} = {ac/len(ap)*100:.1f}%")

print("\nPor familia:")
pf = defaultdict(lambda: defaultdict(list))
for r in rows:
    for e in ["acierto_cifra", "acierto_cita", "acierto_trayectoria"]:
        if r[e] is not None:
            pf[r["familia"]][e].append(r[e])
for f, evs in sorted(pf.items()):
    print(f"  {f}:")
    for e, v in evs.items():
        print(f"    {e}: {sum(v)}/{len(v)}")