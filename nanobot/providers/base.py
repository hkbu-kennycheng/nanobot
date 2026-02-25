"""Base LLM provider interface."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import json_repair


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1 etc.
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Implementations should handle the specifics of each provider's API
    while maintaining a consistent interface.
    """
    
    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace empty text content that causes provider 400 errors.

        Empty content can appear when MCP tools return nothing. Most providers
        reject empty-string content or empty text blocks in list content.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")

            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = None if (msg.get("role") == "assistant" and msg.get("tool_calls")) else "(empty)"
                result.append(clean)
                continue

            if isinstance(content, list):
                filtered = [
                    item for item in content
                    if not (
                        isinstance(item, dict)
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    )
                ]
                if len(filtered) != len(content):
                    clean = dict(msg)
                    if filtered:
                        clean["content"] = filtered
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = None
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue

            result.append(msg)
        return result

    @staticmethod
    def _parse_tool_calls_from_content(
        content: str | None,
    ) -> tuple[str | None, list[ToolCallRequest]]:
        """Extract tool calls embedded as text in content.

        Some models return tool calls as a JSON snippet inside the content
        field instead of populating the structured ``tool_calls`` field.

        Supported formats:

        1. ``Tool Calls:`` prefix followed by a JSON array::

            Tool Calls: [
              {
                "id": "call_...",
                "type": "function",
                "function": { "name": "...", "arguments": "..." }
              }
            ]

        2. Inline JSON object(s) with ``name`` and ``arguments`` keys::

            Some text before the call.
            {"name": "exec", "arguments": {"command": "ls"}}

        Returns:
            ``(remaining_content, tool_calls)`` — the text content with the
            tool-call block removed and the parsed list of
            :class:`ToolCallRequest` objects.  If nothing is detected the
            original *content* is returned unchanged with an empty list.
        """
        if not content:
            return content, []

        # Format 1: "Tool Calls: [...]"
        match = re.search(r"Tool Calls:\s*(\[.*)", content, re.DOTALL)
        if match:
            result = LLMProvider._parse_tool_calls_array(match.group(1))
            if result:
                remaining = content[: match.start()].strip()
                return remaining or None, result

        # Format 2: inline JSON objects with "name" and "arguments"
        inline_results = LLMProvider._parse_inline_tool_calls(content)
        if inline_results:
            remaining, tool_calls = inline_results
            return remaining, tool_calls

        return content, []

    @staticmethod
    def _parse_tool_calls_array(json_str: str) -> list[ToolCallRequest]:
        """Parse a JSON array of tool calls in the OpenAI structure."""
        try:
            raw_calls = json_repair.loads(json_str)
        except Exception:
            return []

        if not isinstance(raw_calls, list):
            return []

        tool_calls: list[ToolCallRequest] = []
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            func = call.get("function")
            if not isinstance(func, dict):
                continue
            name = func.get("name")
            if not name:
                continue

            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json_repair.loads(args)
                except Exception:
                    args = {"raw": args}

            call_id = call.get("id") or f"content_tc_{len(tool_calls)}"
            tool_calls.append(
                ToolCallRequest(
                    id=call_id,
                    name=name,
                    arguments=args if isinstance(args, dict) else {},
                )
            )
        return tool_calls

    @staticmethod
    def _extract_json_object(text: str, start: int) -> str | None:
        """Extract a complete JSON object from *text* beginning at *start*.

        Uses brace-counting (aware of JSON string literals) so that nested
        objects like ``{"name": "x", "arguments": {"a": "b"}}`` are captured
        correctly.
        """
        if start >= len(text) or text[start] != "{":
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @staticmethod
    def _parse_inline_tool_calls(
        content: str,
    ) -> tuple[str | None, list[ToolCallRequest]] | None:
        """Parse inline JSON tool calls like ``{"name": "...", "arguments": {...}}``.

        Scans for JSON objects that contain both ``name`` (str) and
        ``arguments`` keys.  Multiple inline calls in the same content are
        supported.  Handles nested braces correctly via brace-counting.
        """
        tool_calls: list[ToolCallRequest] = []
        # Track spans to remove from content (start, end)
        spans: list[tuple[int, int]] = []

        # Find candidate positions where a JSON object containing "name" starts
        for m in re.finditer(r'\{\s*"name"\s*:', content):
            raw = LLMProvider._extract_json_object(content, m.start())
            if raw is None:
                continue
            try:
                obj = json_repair.loads(raw)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            name = obj.get("name")
            if not isinstance(name, str) or not name:
                continue
            if "arguments" not in obj:
                continue

            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json_repair.loads(args)
                except Exception:
                    args = {"raw": args}

            tool_calls.append(
                ToolCallRequest(
                    id=f"content_tc_{len(tool_calls)}",
                    name=name,
                    arguments=args if isinstance(args, dict) else {},
                )
            )
            spans.append((m.start(), m.start() + len(raw)))

        if not tool_calls:
            return None

        # Remove matched JSON spans from content
        remaining_parts: list[str] = []
        prev_end = 0
        for start, end in spans:
            remaining_parts.append(content[prev_end:start])
            prev_end = end
        remaining_parts.append(content[prev_end:])
        remaining = "".join(remaining_parts).strip()

        return remaining or None, tool_calls

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            model: Model identifier (provider-specific).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
