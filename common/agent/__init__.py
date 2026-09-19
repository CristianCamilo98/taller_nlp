"""API perezosa del agente; importarla no requiere SDK, clave ni dataset."""

__all__ = ["RespuestaFinanciera", "crear_agente", "MODELO", "SYSTEM"]


def __getattr__(name: str):
    if name == "RespuestaFinanciera":
        from .schema import RespuestaFinanciera
        return RespuestaFinanciera
    if name in {"crear_agente", "MODELO", "SYSTEM"}:
        from . import agent
        return getattr(agent, name)
    raise AttributeError(name)
