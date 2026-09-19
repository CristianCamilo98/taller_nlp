"""Función evaluar(ruta_jsonl) — corre el agente sobre un golden set."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from responder import responder
from eval.evaluador_cifra import evaluar_cifra
from eval.evaluador_cita import evaluar_cita
from eval.evaluador_trayectoria import evaluar_trayectoria


def cargar_golden(ruta: str) -> list[dict]:
    return [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]


def evaluar(ruta_jsonl: str, guardar_en: str | None = None) -> list[dict]:
    """Corre responder() sobre cada pregunta y devuelve los resultados."""
    preguntas = cargar_golden(ruta_jsonl)
    resultados = []

    for p in preguntas:
        print(f"[{p['id']}] {p['pregunta'][:80]}...")

        t0 = time.time()
        try:
            respuesta = responder(p["pregunta"])
            error = None
        except Exception as e:
            respuesta = {}
            error = f"{type(e).__name__}: {e}"
        latencia = time.time() - t0

        fila = {
            "id": p["id"],
            "familia": p["familia"],
            "pregunta": p["pregunta"],
            "respuesta_agente": respuesta.get("respuesta"),
            "cifra_agente": respuesta.get("cifra"),
            "unidad_agente": respuesta.get("unidad"),
            "fuente_agente": respuesta.get("fuente"),
            "cita_agente": respuesta.get("cita"),
            "chunk_id_agente": respuesta.get("chunk_id"),
            "tool_calls_agente": respuesta.get("tool_calls_agente"),
            "latencia_s": round(latencia, 2),
            "error": error,
            "respuesta_esperada": p.get("respuesta_esperada"),
            "cifra_esperada": p.get("cifra_esperada"),
        }

        # Aplicar los 3 evaluadores
        fila["acierto_cifra"] = evaluar_cifra(fila, p)
        fila["acierto_cita"] = evaluar_cita(fila, p)
        fila["acierto_trayectoria"] = evaluar_trayectoria(fila, p)

        resultados.append(fila)

    if guardar_en:
        Path(guardar_en).parent.mkdir(parents=True, exist_ok=True)
        with open(guardar_en, "w", encoding="utf-8", newline="\n") as f:
            for r in resultados:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nGuardado en {guardar_en}")

    return resultados


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python evaluar.py <ruta_al_golden.jsonl> [salida.jsonl]")
        sys.exit(1)
    salida = sys.argv[2] if len(sys.argv) > 2 else None
    evaluar(sys.argv[1], salida)