"""Bloquea tool calls idénticas sin conservar estado entre preguntas."""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


DUPLICATE_TOOL_CALL_TYPE = "duplicate_tool_call"


def canonical_tool_call(tool_call: dict[str, Any]) -> tuple[str, str]:
    """Clave estable: nombre y argumentos JSON con claves ordenadas."""
    name = str(tool_call.get("name") or "")
    args = tool_call.get("args") or {}
    normalized = json.dumps(
        args,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return name, normalized


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None) or []
    return [call for call in calls if isinstance(call, dict)]


def is_duplicate_tool_call(
    tool_call: dict[str, Any], messages: list[Any]
) -> bool:
    """Detecta una llamada previa o una duplicada anterior en el mismo lote."""
    wanted = canonical_tool_call(tool_call)
    current_id = tool_call.get("id")
    for message in messages:
        for previous in _message_tool_calls(message):
            if previous.get("id") == current_id:
                break
            if canonical_tool_call(previous) == wanted:
                return True
    return False


class ToolCallDedupMiddleware(AgentMiddleware):
    """Middleware stateless; el historial del run delimita la deduplicación."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        messages = list((request.state or {}).get("messages", []))
        if not is_duplicate_tool_call(request.tool_call, messages):
            return handler(request)
        name, normalized_args = canonical_tool_call(request.tool_call)
        diagnostic = {
            "type": DUPLICATE_TOOL_CALL_TYPE,
            "executed": False,
            "tool_name": name,
            "normalized_args": json.loads(normalized_args),
            "message": "Esta llamada idéntica ya fue realizada en esta pregunta.",
        }
        return ToolMessage(
            content=json.dumps(diagnostic, ensure_ascii=False, sort_keys=True),
            tool_call_id=str(request.tool_call.get("id") or "unknown"),
            name=name or None,
            status="error",
            artifact=diagnostic,
        )
