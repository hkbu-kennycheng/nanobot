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
        The expected format is::

            Tool Calls: [
              {
                "id": "call_...",
                "type": "function",
                "function": { "name": "...", "arguments": "..." }
              }
            ]

        Returns:
            ``(remaining_content, tool_calls)`` — the text content with the
            tool-call block removed and the parsed list of
            :class:`ToolCallRequest` objects.  If nothing is detected the
            original *content* is returned unchanged with an empty list.
        """
        if not content:
            return content, []

        match = re.search(r"Tool Calls:\s*(\[.*)", content, re.DOTALL)
        if not match:
            return content, []

        json_str = match.group(1)
        try:
            raw_calls = json_repair.loads(json_str)
        except Exception:
            return content, []

        if not isinstance(raw_calls, list):
            return content, []

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

        if not tool_calls:
            return content, []

        remaining = content[: match.start()].strip()
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
