"""Re-evalúa un JSONL existente sin llamar al LLM."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.evaluador_cifra import evaluar_cifra
from eval.evaluador_cita import evaluar_cita
from eval.evaluador_trayectoria import evaluar_trayectoria

raiz = Path(__file__).resolve().parents[2]
ruta_resultados = raiz / "marco" / "results" / "baseline" / "resultados_baseline_grupo3.jsonl"
ruta_golden = raiz / "marco" / "golden_set" / "golden_set_grupo3.jsonl"

# Cargar golden para tener los datos de la pregunta (ancla, cifra_esperada, etc.)
golden = {json.loads(l)["id"]: json.loads(l) for l in open(ruta_golden, encoding="utf-8") if l.strip()}

# Re-evaluar
rows = [json.loads(l) for l in open(ruta_resultados, encoding="utf-8") if l.strip()]
for r in rows:
    p = golden[r["id"]]
    r["acierto_cifra"] = evaluar_cifra(r, p)
    r["acierto_cita"] = evaluar_cita(r, p)
    r["acierto_trayectoria"] = evaluar_trayectoria(r, p)

# Guardar el fichero con los evaluadores corregidos
salida = raiz / "marco" / "results" / "baseline" / "resultados_baseline_grupo3_v2.jsonl"
with open(salida, "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Resumen
from collections import defaultdict
for eval_name in ["acierto_cifra", "acierto_cita", "acierto_trayectoria"]:
    aplicables = [r for r in rows if r[eval_name] is not None]
    aciertos = sum(1 for r in aplicables if r[eval_name])
    total = len(aplicables)
    pct = aciertos / total * 100 if total else 0
    print(f"{eval_name}: {aciertos}/{total} = {pct:.1f}%")

print("\nPor familia:")
por_familia = defaultdict(lambda: defaultdict(list))
for r in rows:
    for e in ["acierto_cifra", "acierto_cita", "acierto_trayectoria"]:
        if r[e] is not None:
            por_familia[r["familia"]][e].append(r[e])
for fam, evals in sorted(por_familia.items()):
    print(f"  {fam}:")
    for e, vals in evals.items():
        print(f"    {e}: {sum(vals)}/{len(vals)}")