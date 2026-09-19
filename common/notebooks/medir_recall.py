"""Mide recall@k de densa vs híbrida sobre las 6 extractivas."""
import json
import re
import sys
import unicodedata
from pathlib import Path

raiz = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(raiz / "marco" / "starter"))
sys.path.insert(0, str(raiz / "marco" / "retrieval"))
import miax_s1
from hybrid_bm25 import buscar_hibrido
from query_rewriting import reescribir
import pandas as pd


chunks_meta = pd.read_parquet(raiz / "corpus" / "indice" / "chunks_meta.parquet")


def _norm(t):
    if not t:
        return ""
    t = t.lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


golden = [json.loads(l) for l in open(raiz / "marco" / "golden_set" / "golden_set_grupo3.jsonl", encoding="utf-8") if l.strip()]
extractivas = [p for p in golden if p["familia"] == "extractiva"]


def recall_at_k(resultados, ancla, k):
    """Comprueba si el ancla aparece en alguno de los k primeros chunks."""
    ancla_norm = _norm(ancla)
    for r in resultados[:k]:
        if ancla_norm in _norm(r["texto"]):
            return True
    return False



for k in [1, 3, 5]:
    aciertos_densa = 0
    aciertos_hibrida = 0
    for p in extractivas:
        query = p["pregunta"]  # usar la pregunta como query (se puede mejorar)
        ancla = p["ancla_texto"]
        query_es = p["pregunta"]
        query_en = reescribir(query_es)
        ticker = p["ticker"]
        fy = p["fiscal_year"]
        item = p.get("item_esperado")

        densa = miax_s1.buscar(query_en, ticker=ticker, fiscal_year=fy, item=item, k=k)
        hibrida = buscar_hibrido(query_en, ticker=ticker, fiscal_year=fy, item=item, k=k)

        if recall_at_k(densa, ancla, k): aciertos_densa += 1
        if recall_at_k(hibrida, ancla, k): aciertos_hibrida += 1

    print(f"k={k}: densa={aciertos_densa}/{len(extractivas)}, hibrida={aciertos_hibrida}/{len(extractivas)}")