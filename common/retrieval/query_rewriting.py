"""Query rewriting: traduce la pregunta al inglés antes de buscar.

El corpus está en inglés. Si la pregunta viene en español, la búsqueda
densa funciona a medias (BGE es multilingüe) y BM25 no funciona en
absoluto. Traducir la query al inglés mejora ambas.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Cargar .env de la raíz
_raiz = Path(__file__).resolve().parents[2]
for ruta in [_raiz / ".env", _raiz / "marco" / ".env"]:
    if ruta.is_file():
        load_dotenv(ruta)
        break

_MODELO = "openrouter:google/gemini-3.5-flash-lite"  # baratísimo
_modelo = None


def _get_modelo():
    global _modelo
    if _modelo is None:
        _modelo = init_chat_model(_MODELO, temperature=0)
    return _modelo


PROMPT = """Translate the following question to English for use as a
search query over English 10-K financial reports. Return ONLY the
translation, no explanations, no quotes.

Question (in Spanish): {pregunta}

English query:"""


def reescribir(pregunta: str) -> str:
    """Traduce la pregunta al inglés para usarla como query de búsqueda."""
    try:
        modelo = _get_modelo()
        respuesta = modelo.invoke(PROMPT.format(pregunta=pregunta))
        return respuesta.text.strip().strip('"').strip("'")
    except Exception as e:
        print(f"⚠️ reescribir falló: {type(e).__name__}: {e}")
        return pregunta  # fallback: usa la original


if __name__ == "__main__":
    ejemplos = [
        "¿Qué políticas contables relevantes describe Amazon en sus estados financieros de FY2024?",
        "¿Qué riesgo regulatorio relacionado con privacidad de datos menciona Meta en sus factores de riesgo de FY2024?",
        "¿Qué dice Microsoft sobre su exposición al riesgo de tipo de cambio en el apartado de riesgo de mercado de FY2024?",
    ]
    for p in ejemplos:
        print(f"ES: {p}")
        print(f"EN: {reescribir(p)}")
        print()