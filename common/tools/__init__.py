"""Acceso perezoso a las cuatro tools públicas."""

__all__ = ["list_available", "get_xbrl_fact", "search_filings", "read_section"]


def __getattr__(name: str):
    if name in __all__:
        from . import tools
        return getattr(tools, name)
    raise AttributeError(name)
