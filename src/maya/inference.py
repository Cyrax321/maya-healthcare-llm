"""Repeatable prediction interface and model-output parsing.

Parsing is pure Python and CPU-safe so it is unit-testable without a GPU. It
handles the Qwen3.5 reasoning-format tokens (`` thinking`` / `` response``), the
native function-calling block format, JSON extraction, label and numeric
extraction, and abstention detection.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from .metrics import extract_numbers, normalize_text


@dataclass
class ParsedOutput:
    raw: str
    response: str
    tool_calls: list[dict[str, Any]] | None = None
    json: dict[str, Any] | None = None
    label: str | None = None
    numeric: float | None = None


class Generator(Protocol):
    """Anything callable that maps messages -> generated text."""

    def __call__(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        ...


def strip_reasoning(text: str) -> str:
    """Return the assistant's visible answer, dropping the `` thinking`` block.

    The Qwen3.5 reasoning format is::

         thinking
        <hidden reasoning>
         response

        <answer>
    """
    if not text:
        return ""
    if " response" in text:
        return text.split(" response", 1)[1].lstrip("\n")
    if " thinking" in text:
        return text.split(" thinking", 1)[1].lstrip("\n")
    return text


_TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>")
_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _coerce_value(raw: str) -> Any:
    raw = raw.strip()
    # JSON first (handles mappings/sequences emitted via tojson)
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        pass
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """Parse Qwen3.5 native ``<tool_call>`` blocks into structured calls."""
    blocks = _TOOL_BLOCK_RE.findall(text)
    if not blocks:
        return None
    calls: list[dict[str, Any]] = []
    for block in blocks:
        fn_match = _FUNCTION_RE.search(block)
        if not fn_match:
            continue
        name = fn_match.group(1).strip()
        arguments: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(block):
            key = pm.group(1).strip()
            val = _coerce_value(pm.group(2))
            arguments[key] = val
        calls.append({"name": name, "arguments": arguments})
    return calls or None


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from free text, if any."""
    if not text:
        return None
    stripped = text.strip()
    try:
        val = json.loads(stripped)
        if isinstance(val, dict):
            return val
    except (ValueError, TypeError):
        pass
    # fall back to brace-matching extraction
    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(stripped)):
        c = stripped[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    val = json.loads(stripped[start : i + 1])
                    return val if isinstance(val, dict) else None
                except (ValueError, TypeError):
                    return None
    return None


_ABSTAIN_PATTERNS = (
    "cannot answer",
    "not enough information",
    "no information",
    "insufficient information",
    "unable to answer",
    "do not know",
    "i don't know",
    "not provided",
    "no policy",
)


def is_abstention(response: str) -> bool:
    r = normalize_text(response)
    return any(p in r for p in _ABSTAIN_PATTERNS)


def extract_label(response: str, labels: Sequence[str]) -> str | None:
    r = normalize_text(response)
    for lab in labels:
        if normalize_text(lab) in r:
            return lab
    return None


def parse_completion(text: str) -> ParsedOutput:
    response = strip_reasoning(text)
    return ParsedOutput(
        raw=text,
        response=response,
        tool_calls=parse_tool_calls(text),
        json=extract_json(response),
        label=None,
        numeric=(extract_numbers(response) or [None])[0],
    )


def build_pred(
    text: str,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Turn raw generated text into a structured prediction dict.

    ``pred`` shape matches the canonical result documented in ``metrics``.
    """
    parsed = parse_completion(text)
    label = None
    if labels:
        label = extract_label(parsed.response, labels)
    return {
        "text": parsed.response,
        "label": label,
        "numeric": parsed.numeric,
        "json": parsed.json,
        "tool_calls": parsed.tool_calls,
        "abstain": is_abstention(parsed.response),
    }


class StubGenerator:
    """Deterministic offline generator used only to smoke-test the pipeline.

    It echoes a canned answer (optionally the ground truth) so the metric and
    artifact machinery can be validated end-to-end without a GPU. Artifacts
    produced from this stub are never reported as real model performance.
    """

    def __init__(self, mode: str = "echo"):
        self.mode = mode

    def __call__(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if self.mode == "echo":
            return f" thinking\nstub reasoning\n response\n\n{user}"
        return " thinking\nstub reasoning\n response\n\nok"
