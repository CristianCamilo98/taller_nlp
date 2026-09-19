"""Baseline común para el agente 10-K.

El paquete no importa componentes pesados de forma implícita. En particular,
``import common`` no carga datasets, modelos de embeddings ni clientes LLM.
"""

__version__ = "2.0.0-dev"

__all__ = ["__version__"]
