"""Turno de reparación de citación: repara chunk_id/cita SOLO desde chunks
realmente recuperados por search_filings, nunca inventa evidencia.

Un smoke extractivo real (con routing y argumentos ya corregidos en la
iteración anterior de esta auditoría) mostró un fallo distinto:
acierto_trayectoria=True -la llamada a search_filings fue exactamente la
esperada, con ticker/fiscal_year/item/k correctos- pero la respuesta final
no citó ningún chunk: fuente="ninguna", respuesta="No se encontró
información en el corpus.". Esto es DISTINTO del hueco de XBRL: el system
prompt del agente permite explícitamente declarar que no hay evidencia
suficiente ("Si el dato no está en el corpus, dilo. No lo estimes."), así
que fuente="ninguna" sin cita es, en principio, una respuesta legítima, no
un defecto mecánico.

Este módulo por eso solo actúa cuando la respuesta SÍ declara evidencia
textual (fuente en {"texto", "ambas"}) pero cita/chunk_id están vacíos o no
corresponden a un chunk realmente recuperado -ese caso sí es análogo al de
XBRL: el modelo dice haber usado el filing pero no lo tradujo a los campos
estructurados-.

citation_repair_turn() pide al modelo, dado el conjunto EXACTO de chunks
recuperados y la respuesta ya generada, elegir un chunk_id y extraer una
cita literal de él, sin cambiar el contenido de la respuesta ni inventar
texto. El resultado se valida DETERMINÍSTICAMENTE después de la llamada,
nunca se confía en el modelo: chunk_id debe estar entre los chunks
realmente recuperados y la cita debe estar contenida literalmente
(normalizada, con soporte para omisiones "...") en el texto de ESE chunk.
Si algo no cumple, se descarta (None) y el llamador falla cerrado.

Un smoke extractivo posterior, ya con tool_results_detallado capturando los
chunks reales, mostró un tercer caso: retrieval correcto (el chunk relevante
estaba en el top-5, verificado contra retrieval-final-v1 -mismos IDs, mismo
orden, mismos scores-) y ese chunk contenía literalmente la evidencia
necesaria, pero el modelo declaró fuente="ninguna" sin citar nada. Esto ya
no es un defecto mecánico de extracción (como el de XBRL o el de
citation_repair_turn); es que el modelo, con evidencia real disponible, no
la reconoció como suficiente. evidence_adjudication_turn() cubre este
tercer caso: cuando fuente="ninguna" pero search_filings sí recuperó
chunks, pide a un modelo aparte que revise SOLO esos chunks y decida si
alguno responde la pregunta, reconstruyendo la respuesta exclusivamente
desde ese chunk si es así. La respuesta previa (la declinación) se pasa
solo como contexto, nunca como verdad. Igual que citation_repair_turn, el
resultado se valida determinísticamente: chunk_id real + cita literal en
ese chunk + respuesta no vacía y no otra declinación genérica. Si algo no
cumple, o el modelo confirma que no hay evidencia suficiente, se mantiene
la declinación original -nunca se fuerza ni se inventa una respuesta-.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

CITATION_REPAIR_SYSTEM = """Se te da una respuesta en prosa ya generada y
los fragmentos (chunks) REALMENTE recuperados de los informes 10-K para
esta pregunta. Elige el chunk que mejor respalda esa respuesta y copia de
él una cita literal breve -puedes usar "..." para omitir texto intermedio,
pero cada fragmento debe ser una copia exacta del chunk, nunca una
paráfrasis-. No cambies el contenido de la respuesta ni inventes texto que
no esté en el chunk elegido. Si ningún chunk respalda la respuesta,
responde con respaldado=false y deja chunk_id/cita vacíos."""


class CitationRepairResult(BaseModel):
    respaldado: bool = Field(
        description=("True solo si alguno de los chunks dados respalda la "
                     "respuesta ya generada"))
    chunk_id: str | None = Field(
        default=None,
        description="chunk_id elegido, tal cual aparece en la lista dada")
    cita: str | None = Field(
        default=None,
        description="Fragmento literal copiado del chunk elegido")


def _normalizar(text: str) -> str:
    value = "".join(ch for ch in unicodedata.normalize("NFD", str(text))
                    if unicodedata.category(ch) != "Mn").lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _fragments(cita: str) -> list[str]:
    return [part for part in (_normalizar(p) for p in
            re.split(r"\s*(?:\.\.\.|…)+\s*", cita)) if part]


def citation_in_text(cita: str | None, texto: str) -> bool:
    """Cita literal (normalizada; admite omisiones "...") dentro del texto."""
    if not cita or not str(cita).strip():
        return False
    pieces = _fragments(cita)
    normalized_text = _normalizar(texto)
    return bool(pieces) and all(piece in normalized_text for piece in pieces)


_model = None


def _get_citation_repair_model():
    global _model
    if _model is None:
        from common.agent.agent import crear_forced_evidence_model
        _model = crear_forced_evidence_model()
    return _model


def citation_repair_turn(
    pregunta: str,
    respuesta: str,
    chunks: list[dict],
    *,
    model_fn: Callable[[], Any] | None = None,
) -> tuple[str, str] | None:
    """Un único turno; None si no hay reparación válida (fail closed).

    ``chunks`` debe contener solo chunks realmente devueltos por
    search_filings en esta ejecución. Nunca se acepta un chunk_id fuera de
    esa lista ni una cita no contenida literalmente en su texto -eso se
    comprueba aquí, después de la llamada; no se confía en el modelo-.
    """
    if not chunks:
        return None

    model = (model_fn or _get_citation_repair_model)()
    structured_model = model.with_structured_output(CitationRepairResult)
    listado = "\n\n---\n\n".join(
        f"[{chunk['chunk_id']}]\n{chunk['texto']}" for chunk in chunks
    )
    prompt = (
        f"Pregunta original: {pregunta}\n\n"
        f"Respuesta ya generada: {respuesta}\n\n"
        f"Chunks realmente recuperados:\n\n{listado}"
    )
    result = structured_model.invoke([
        SystemMessage(content=CITATION_REPAIR_SYSTEM),
        HumanMessage(content=prompt),
    ])
    if not getattr(result, "respaldado", False):
        return None
    chunk_id = getattr(result, "chunk_id", None)
    cita = getattr(result, "cita", None)
    if not chunk_id or not cita:
        return None

    matching = next((c for c in chunks if c["chunk_id"] == chunk_id), None)
    if matching is None:
        return None
    if not citation_in_text(cita, matching["texto"]):
        return None
    return chunk_id, cita


EVIDENCE_ADJUDICATION_SYSTEM = """Se te da una pregunta financiera y los
fragmentos (chunks) REALMENTE recuperados de los informes 10-K para esa
pregunta. Determina si alguno de esos chunks contiene evidencia suficiente
para responderla.

- Si SÍ hay evidencia suficiente en al menos un chunk: responde
  respaldado=true, sintetiza una respuesta breve BASADA ÚNICAMENTE en el
  contenido de ese chunk -nunca en conocimiento externo ni en otro
  documento-, elige el chunk_id que la respalda y copia de él una cita
  literal breve -puedes usar "..." para omitir texto intermedio, pero cada
  fragmento debe ser una copia exacta, nunca una paráfrasis-.
- Si NINGÚN chunk dado responde realmente a la pregunta: responde
  respaldado=false y deja respuesta/chunk_id/cita vacíos. No inventes ni
  fuerces una respuesta; declinar es la respuesta correcta cuando no hay
  evidencia suficiente."""


class EvidenceAdjudicationResult(BaseModel):
    respaldado: bool = Field(
        description=("True solo si al menos uno de los chunks dados "
                     "contiene evidencia suficiente para responder"))
    respuesta: str | None = Field(
        default=None,
        description="Respuesta sintetizada solo desde el chunk elegido, "
                    "o null si respaldado=false")
    chunk_id: str | None = Field(
        default=None,
        description="chunk_id elegido, tal cual aparece en la lista dada")
    cita: str | None = Field(
        default=None,
        description="Fragmento literal copiado del chunk elegido")


_DECLINE_MARKERS = (
    "ninguna", "no encontrado", "no encontre", "no se encontro",
    "no hay informacion", "no hay evidencia", "sin informacion",
    "sin evidencia", "no disponible", "not found", "no information",
    "no se encuentra",
)


def _looks_like_decline(respuesta: str) -> bool:
    """Detecta declinaciones genericas ("ninguna", "no encontrado", ...),
    nunca vocabulario de una pregunta concreta."""
    normalized = _normalizar(respuesta)
    if len(normalized) < 8:
        return True
    return any(marker in normalized for marker in _DECLINE_MARKERS)


def evidence_adjudication_turn(
    pregunta: str,
    respuesta_previa: str,
    chunks: list[dict],
    *,
    model_fn: Callable[[], Any] | None = None,
) -> tuple[str, str, str] | None:
    """Un único turno; None si no hay evidencia suficiente o la reparación
    no pasa validación determinista (fail closed -nunca fuerza respuesta).

    Devuelve (respuesta_reparada, chunk_id, cita) SOLO si:
    - el modelo declara respaldado=true;
    - chunk_id está entre los chunks realmente recuperados;
    - cita está contenida literalmente en el texto de ESE chunk;
    - respuesta no está vacía ni es una declinación genérica equivalente.

    ``respuesta_previa`` se da solo como contexto -nunca se valida contra
    ella, no se confía en que sea correcta-.
    """
    if not chunks:
        return None

    model = (model_fn or _get_citation_repair_model)()
    structured_model = model.with_structured_output(EvidenceAdjudicationResult)
    listado = "\n\n---\n\n".join(
        f"[{chunk['chunk_id']}]\n{chunk['texto']}" for chunk in chunks
    )
    prompt = (
        f"Pregunta original: {pregunta}\n\n"
        "Respuesta previa (contexto, no la des por buena; decide de nuevo "
        f"solo a partir de los chunks): {respuesta_previa}\n\n"
        f"Chunks realmente recuperados:\n\n{listado}"
    )
    result = structured_model.invoke([
        SystemMessage(content=EVIDENCE_ADJUDICATION_SYSTEM),
        HumanMessage(content=prompt),
    ])
    if not getattr(result, "respaldado", False):
        return None
    respuesta = getattr(result, "respuesta", None)
    chunk_id = getattr(result, "chunk_id", None)
    cita = getattr(result, "cita", None)
    if not respuesta or not respuesta.strip():
        return None
    if _looks_like_decline(respuesta):
        return None
    if not chunk_id or not cita:
        return None

    matching = next((c for c in chunks if c["chunk_id"] == chunk_id), None)
    if matching is None:
        return None
    if not citation_in_text(cita, matching["texto"]):
        return None
    return respuesta.strip(), chunk_id, cita
