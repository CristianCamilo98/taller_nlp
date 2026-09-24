"""Módulo auxiliar de la sesión 1 — lo que hoy es caja negra.

Se reparte junto a `S1_Herramientas_y_Bucle_*.ipynb`. Contiene dos cosas, y
las dos están aquí por el mismo motivo: **no son el contenido de la sesión 1**.

1. `buscar()` — la búsqueda densa sobre el índice FAISS que hay debajo de la
   herramienta `search_filings`. Es caja negra a propósito hasta el día 17,
   que es cuando se abre y se cuestiona (troceado, embeddings, `IndexFlatIP`,
   top-k, y los tres arreglos que se miden con `recall@k`).

2. `demo_apertura()` — la demo del minuto uno. Reproduce una ejecución del
   agente terminado, grabada en `demo_traza.json`, para enseñar el destino
   antes de construirlo. **No depende ni de la red ni de que el corpus esté
   montado**: en el minuto uno de clase todavía no se ha ejecutado la celda de
   instalación ni la de setup.

## Lo único que hay que saber de `buscar()`

El modelo lo elige ``SETTINGS.embedding_model``. Si es BGE local, la consulta
lleva el prefijo documentado en ``indice/MANIFEST.md``. Si es OpenRouter
(p. ej. ``google/gemini-embedding-2``), no hay prefijo: se usa ``input_type``
``search_query`` / ``search_document``. Mezclar un FAISS de un modelo con
el encoder de otro no da un error obvio de tipos: recupera basura.
"""

from __future__ import annotations

import functools
import json
import threading
from pathlib import Path

from cristian.experiments import config as config_mod
from cristian.experiments.config import (
    embedding_index_path,
    embedding_manifest_path,
    embedding_model_id,
    get_dataset_paths,
    is_openrouter_embedding,
)

RUTA_TRAZA_DEMO = Path(__file__).resolve().parent / "demo_traza.json"

_indice_lock = threading.Lock()


def _mensaje_indice_ausente(model: str, index_path: Path) -> str:
    return (
        f"No hay índice FAISS para {model} en {index_path}. "
        "Genera primero: python -m cristian.experiments.indexar"
    )


# ---------------------------------------------------------------------------
# Búsqueda densa — el cuerpo de `search_filings`
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def _cargar_indice_para(model: str):
    """Carga FAISS + encoder del ``embedding_model`` (una vez por modelo)."""
    import faiss
    import pandas as pd

    from cristian.experiments.embeddings import crear_encoder

    paths = get_dataset_paths()
    index_path = embedding_index_path(model)
    if not index_path.is_file():
        raise RuntimeError(_mensaje_indice_ausente(model, index_path))

    indice = faiss.read_index(str(index_path))
    meta = pd.read_parquet(paths.chunks_meta)

    if indice.ntotal != len(meta):
        raise RuntimeError(
            f"El índice tiene {indice.ntotal} vectores y los metadatos "
            f"{len(meta)} filas. Están desalineados."
        )

    if is_openrouter_embedding(model):
        manifest_path = embedding_manifest_path(model)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = embedding_model_id(model)
            stored = manifest.get("model")
            if stored and stored != expected:
                raise RuntimeError(
                    f"El manifiesto de {index_path} es del modelo {stored}, "
                    f"pero SETTINGS pide {expected}. Regenera con "
                    "python -m cristian.experiments.indexar --force"
                )

    encoder = crear_encoder(model)
    return indice, meta, encoder


def _indice():
    """Índice, metadatos y encoder compartidos entre hilos.

    ``lru_cache`` solo memoiza el resultado: si varios workers entran a la
    vez antes de que termine la primera carga, todos recargarían. El lock
    serializa esa primera carga.
    """
    with _indice_lock:
        return _cargar_indice_para(config_mod.SETTINGS.embedding_model)

def precargar_retrieval() -> None:
    """Fuerza la carga de FAISS + encoder en el hilo principal."""
    _indice()


def clear_retrieval_cache() -> None:
    _cargar_indice_para.cache_clear()


def buscar(
    query: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    k: int = 5,
) -> list[dict]:
    """Los `k` fragmentos más parecidos a `query`, con sus metadatos.

    Cada resultado es un dict con `chunk_id`, `ticker`, `fiscal_year`, `item`,
    `texto`, `n_tokens`, `contiene_tabla` y `puntuacion` (similitud coseno,
    entre -1 y 1: los vectores están normalizados y el índice usa producto
    interno).

    Los filtros se aplican **después** de la búsqueda, sobre el orden que
    devuelve el índice. Con 1.749 vectores eso es instantáneo y no cambia el
    resultado; con un corpus grande habría que filtrar antes, y esa es una de
    las conversaciones del día 17.
    """
    indice, meta, encoder = _indice()

    vector = encoder.encode_query(query)
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    if vector.shape[1] != indice.d:
        raise RuntimeError(
            f"La query tiene dim {vector.shape[1]} y el índice {indice.d}. "
            "El embedding_model de SETTINGS no coincide con el FAISS cargado. "
            "Regenera: python -m cristian.experiments.indexar --force"
        )

    puntuaciones, posiciones = indice.search(vector, indice.ntotal)

    resultados: list[dict] = []
    for puntuacion, posicion in zip(puntuaciones[0], posiciones[0]):
        fila = meta.iloc[int(posicion)]
        # Ojo: `fila.item` devuelve el método `Series.item`, no la columna.
        # Con corchetes siempre.
        if ticker and fila["ticker"] != ticker:
            continue
        if fiscal_year and int(fila["fiscal_year"]) != int(fiscal_year):
            continue
        if item and fila["item"] != item:
            continue
        resultados.append(
            {
                "chunk_id": fila["chunk_id"],
                "ticker": fila["ticker"],
                "fiscal_year": int(fila["fiscal_year"]),
                "item": fila["item"],
                "texto": fila["texto"],
                "n_tokens": int(fila["n_tokens"]),
                "contiene_tabla": bool(fila["contiene_tabla"]),
                "puntuacion": round(float(puntuacion), 4),
            }
        )
        if len(resultados) >= k:
            break
    return resultados


def buscar_con_rerank(
    query: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    k: int = 5,
    *,
    pool_k: int | None = None,
    reranker_model: str | None = None,
) -> list[dict]:
    """Dense (SETTINGS.embedding_model) → pool → cross-encoder → top-``k``.

    El pool lo saca el mismo FAISS/encoder que ``buscar`` (p. ej. Gemini).
    El reranker solo reordena esos candidatos.
    """
    from cristian.experiments.reranker import rerank_hits

    pool_size = pool_k if pool_k is not None else config_mod.SETTINGS.rerank_pool_k
    model = reranker_model or config_mod.SETTINGS.reranker_model
    pool = buscar(
        query,
        ticker=ticker,
        fiscal_year=fiscal_year,
        item=item,
        k=max(int(pool_size), int(k)),
    )
    return rerank_hits(
        query,
        pool,
        top_k=k,
        model_name=model,
        max_length=config_mod.SETTINGS.rerank_max_length,
    )


def precargar_reranker(model_name: str | None = None) -> None:
    """Precarga el cross-encoder en el hilo principal."""
    from cristian.experiments.reranker import precargar_reranker as _precargar

    _precargar(
        model_name or config_mod.SETTINGS.reranker_model,
        max_length=config_mod.SETTINGS.rerank_max_length,
    )


def formatear_fragmentos(fragmentos: list[dict]) -> str:
    """Los fragmentos, en el texto que ve el modelo.

    Cada uno lleva su `chunk_id` delante: es lo que permite que el agente cite
    y que el evaluador `cita_correcta` compruebe la cita.
    """
    if not fragmentos:
        return ("Sin resultados para esa consulta con esos filtros. "
                "Prueba a quitar algún filtro o a reformular la búsqueda.")
    partes = []
    for f in fragmentos:
        partes.append(
            f"[{f['chunk_id']}] {f['ticker']} FY{f['fiscal_year']} "
            f"Item {f['item']} (similitud {f['puntuacion']:.3f})\n{f['texto']}"
        )
    return "\n\n---\n\n".join(partes)


# ---------------------------------------------------------------------------
# La demo del minuto uno
# ---------------------------------------------------------------------------
def _imprimir_paso(n: int, paso: dict) -> None:
    if paso["tipo"] == "herramienta":
        args = ", ".join(f"{k}={v!r}" for k, v in paso["argumentos"].items())
        print(f"  {n}. {paso['herramienta']}({args})")
        utiles = [x for x in paso["resultado"].splitlines() if x.strip()]
        for linea in utiles[:2]:
            print(f"       -> {linea[:100]}")
    elif paso["tipo"] == "razonamiento":
        print(f"  {n}. (el modelo decide) {paso['texto']}")


def demo_apertura(agente=None, pregunta: str | None = None) -> dict:
    """Enseña el agente terminado antes de construirlo.

    Sin argumentos reproduce la ejecución grabada en `demo_traza.json`, que es
    lo que hace falta en el minuto uno: ahí todavía no hay ni LangChain
    instalado ni corpus montado.

    Si se le pasa un `agente` ya construido —al final de la sesión, con el
    agente de §6— ejecuta la pregunta de verdad y, si algo falla, cae en la
    grabación en vez de romper la clase.
    """
    grabacion = json.loads(RUTA_TRAZA_DEMO.read_text(encoding="utf-8"))
    pregunta = pregunta or grabacion["pregunta"]

    if agente is not None:
        try:
            resultado = agente.invoke(
                {"messages": [{"role": "user", "content": pregunta}]},
                config={"configurable": {"thread_id": "demo-apertura"}},
            )
            print(f"PREGUNTA: {pregunta}\n")
            print("(ejecución en vivo)\n")
            print(resultado["structured_response"].respuesta)
            return resultado
        except Exception as e:                      # la clase no se para
            print(f"No se pudo ejecutar en vivo ({type(e).__name__}: {e}).")
            print("Reproduzco la ejecución grabada.\n")

    print(f"PREGUNTA: {pregunta}\n")
    print("TRAYECTORIA")
    for i, paso in enumerate(grabacion["trayectoria"], 1):
        _imprimir_paso(i, paso)
    print("\nRESPUESTA")
    print(grabacion["respuesta"]["respuesta"])
    print(f"\nfuente: {grabacion['respuesta']['fuente']}"
          f" · cita: {grabacion['respuesta']['chunk_id']}")

    meta = grabacion["metadatos"]
    latencia = (f" · {meta['latencia_s']:.1f} s"
                if meta.get("latencia_s") else "")
    print(f"\n[grabación del {meta['fecha']} · modelo {meta['modelo']}"
          f" · {meta['llamadas_herramienta']} llamadas a herramienta"
          f"{latencia}]")
    if meta.get("origen") != "ejecucion_real":
        print("[AVISO: traza PROVISIONAL. Las salidas de herramienta son "
              "reales, la prosa final es de ejemplo. Regenérala con "
              "`python generar_traza_demo.py` en cuanto haya clave.]")
    return grabacion
