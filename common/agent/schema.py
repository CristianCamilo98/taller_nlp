"""Esquema de respuesta estructurada del agente."""
from typing import Literal

from pydantic import BaseModel, Field


class RespuestaFinanciera(BaseModel):
    """Respuesta trazable a una pregunta sobre informes 10-K."""

    respuesta: str = Field(
        description="Respuesta en prosa, breve y directa")
    cifra: float | None = Field(
        default=None, description="Valor numérico, si la pregunta pide uno")
    unidad: str | None = Field(
        default=None, description="USD, shares, porcentaje…")
    ticker: str | None = None
    ejercicio: int | None = None
    fuente: Literal["xbrl", "texto", "ambas", "ninguna"] = Field(
        description="De dónde sale el dato. 'ninguna' si no está en el corpus")
    cita: str | None = Field(
        default=None,
        description="Texto literal del informe que respalda la respuesta")
    chunk_id: str | None = Field(
        default=None,
        description="Identificador del fragmento citado, para verificar")