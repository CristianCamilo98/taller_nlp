"""Esquema compatible y trazable de la respuesta del agente."""

from typing import Literal

from pydantic import BaseModel, Field


class RespuestaFinanciera(BaseModel):
    """Respuesta sobre 10-K; los campos comparativos son extensiones v2."""

    respuesta: str = Field(description="Respuesta en prosa, breve y directa")
    cifra: float | None = Field(
        default=None,
        description=("Valor pedido: hecho XBRL en numéricas o delta absoluto "
                     "en comparativas"),
    )
    unidad: str | None = Field(default=None, description="Unidad XBRL exacta")
    ticker: str | None = None
    ejercicio: int | None = Field(
        default=None, description="Ejercicio de la pregunta no comparativa")
    fuente: Literal["xbrl", "texto", "ambas", "ninguna"] = Field(
        description="Fuente autorizada utilizada")
    cita: str | None = Field(
        default=None, description="Fragmento literal que respalda la respuesta")
    chunk_id: str | None = Field(
        default=None, description="Identificador del fragmento citado")

    # Extensión compatible necesaria para evaluar comparativas sin interpretar
    # prosa. Los consumidores antiguos pueden ignorar estos campos opcionales.
    concepto_xbrl: str | None = None
    ejercicio_inicial: int | None = None
    ejercicio_final: int | None = None
    valor_inicial: float | None = None
    valor_final: float | None = None
    delta: float | None = None
    porcentaje: float | None = Field(
        default=None, description="Variación porcentual relativa")
