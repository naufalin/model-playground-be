from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

TOOL_NAME_MAX_LENGTH = 100
OUTPUT_PREVIEW_MAX_LENGTH = 500
VISUALIZATION_TOOL_NAME = "generate_visualization"
VISUALIZATION_PAGE_TYPES = {"dashboard", "report", "comparison", "chart"}


@dataclass(frozen=True)
class ChatThreadInfo:
    id: int
    encoded_id: str
    provider: str
    model_name: str
    request_options: dict[str, str]


@dataclass(frozen=True)
class CapturedMessage:
    role: str
    content: str
    latency_ms: int | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_input: dict[str, Any] | None = None
    output_preview: str | None = None
    viz_html: str | None = None
    provider: str | None = None
    model: str | None = None
    usage_json: dict[str, Any] | None = None
    thinking_json: dict[str, Any] | None = None
    request_options_json: dict[str, str] | None = None
    output_delta_count: int | None = None
    selected_skill: str | None = None
    turn_id: str | None = None
    transcript_sequence: int | None = None


class ChatThreadCapture:
    """Capture runtime stream state for one playground thread."""

    def __init__(
        self,
        thread: ChatThreadInfo,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.thread = thread
        self._monotonic = monotonic
        self._start = monotonic()
        self._finished_at: float | None = None
        self._first_token_ms: int | None = None
        self._text = ""
        self._done: dict[str, Any] | None = None
        self._timeline: list[dict[str, Any]] = []
        self._selected_skill: str | None = None
        self._turn_id = str(uuid4())
        self._transcript_text_valid = True

    def start_event(self) -> dict[str, Any]:
        return {
            "type": "thread_start",
            "thread_id": self.thread.encoded_id,
            "provider": self.thread.provider,
            "model": self.thread.model_name,
        }

    def observe(self, event: dict[str, Any]) -> dict[str, Any] | None:
        event_type = event.get("type")
        if event_type == "done":
            self._done = event
            selected_skill = event.get("selected_skill")
            if isinstance(selected_skill, str):
                self._selected_skill = selected_skill
            self._reconcile_transcript_text()
            self._finish()
            return None

        if event_type == "skill_selected":
            skill = event.get("skill")
            if isinstance(skill, str):
                self._selected_skill = skill
                _append_timeline_event(self._timeline, event)

        if event_type == "text_delta":
            delta = event.get("delta", "")
            if isinstance(delta, str):
                if self._first_token_ms is None and delta.strip():
                    self._first_token_ms = int((self._monotonic() - self._start) * 1000)
                self._text += delta
                _append_timeline_event(self._timeline, event)

        if event_type in {"thinking_delta", "tool_start", "tool_end"}:
            _append_timeline_event(self._timeline, event)

        if event_type == "error":
            _append_timeline_event(self._timeline, event)
            self._finish()

        return event

    def error_event(self, exc: Exception) -> dict[str, Any]:
        event = {
            "type": "error",
            "thread_id": self.thread.encoded_id,
            "error": str(exc),
        }
        self.observe(event)
        return event

    def cancel(self) -> None:
        self._done = {
            "provider": self.thread.provider,
            "model": self.thread.model_name,
            "usage": {"cancelled": True},
        }
        self._finish()

    def thread_done_event(self) -> dict[str, Any]:
        self._finish()
        done = self._done or {}
        return {
            "type": "thread_done",
            "thread_id": self.thread.encoded_id,
            "latency_ms": self.latency_ms,
            "content": self.content,
            "provider": done.get("provider") or self.thread.provider,
            "model": done.get("model") or self.thread.model_name,
            "usage": self.usage,
            "thinking": done.get("thinking"),
            "output_delta_count": done.get("output_delta_count"),
            "selected_skill": self._selected_skill,
        }

    @property
    def latency_ms(self) -> int:
        finished_at = self._finished_at if self._finished_at is not None else self._monotonic()
        return int((finished_at - self._start) * 1000)

    @property
    def content(self) -> str:
        done = self._done or {}
        content = done.get("content")
        return content if isinstance(content, str) else self._text

    @property
    def usage(self) -> dict[str, Any] | None:
        done = self._done or {}
        usage = done.get("usage")
        return _usage_with_ttft(usage if isinstance(usage, dict) else None, self._first_token_ms)

    def captured_messages(self) -> list[CapturedMessage]:
        done = self._done or {}
        timeline = list(self._timeline)
        if not any(event.get("type") == "thinking" for event in timeline) and done.get("thinking"):
            thinking_text = _thinking_text(
                done.get("thinking"),
                done.get("provider") or self.thread.provider,
            )
            if thinking_text:
                thinking_event = {
                    "type": "thinking",
                    "kind": _thinking_kind(done.get("provider") or self.thread.provider),
                    "content": thinking_text,
                    "thinking": done.get("thinking"),
                }
                first_text_index = next(
                    (
                        index
                        for index, event in enumerate(timeline)
                        if event.get("type") == "text"
                    ),
                    len(timeline),
                )
                timeline.insert(first_text_index, thinking_event)

        if not self._transcript_text_valid:
            timeline = [event for event in timeline if event.get("type") != "text"]

        messages = [
            _timeline_message(
                event,
                turn_id=self._turn_id,
                transcript_sequence=sequence,
            )
            for sequence, event in enumerate(timeline)
        ]
        content = self.content
        has_canonical_content = isinstance(done.get("content"), str) or bool(content)
        if has_canonical_content:
            messages.append(
                CapturedMessage(
                    role="assistant",
                    content=content,
                    latency_ms=self.latency_ms,
                    provider=done.get("provider") or self.thread.provider,
                    model=done.get("model") or self.thread.model_name,
                    usage_json=self.usage,
                    thinking_json=done.get("thinking"),
                    request_options_json=self.thread.request_options,
                    output_delta_count=done.get("output_delta_count"),
                    selected_skill=self._selected_skill,
                    turn_id=self._turn_id,
                )
            )
        return messages

    def _finish(self) -> None:
        if self._finished_at is None:
            self._finished_at = self._monotonic()

    def _reconcile_transcript_text(self) -> None:
        canonical = self.content
        streamed = "".join(
            str(event.get("content") or "")
            for event in self._timeline
            if event.get("type") == "text"
        )
        if canonical == streamed:
            return
        if canonical.startswith(streamed):
            suffix = canonical[len(streamed) :]
            if suffix:
                _append_timeline_event(
                    self._timeline,
                    {"type": "text_delta", "delta": suffix},
                )
            return
        self._transcript_text_valid = False


def _request_options(
    provider: str,
    model_name: str,
    reasoning_effort: str | None,
) -> dict[str, str]:
    options = {"provider": provider, "model": model_name}
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    return options


def _bounded_text(value: str, max_length: int) -> str:
    return value if len(value) <= max_length else value[:max_length]


def _normalize_tool_name(raw_tool: Any) -> str:
    if not isinstance(raw_tool, str):
        return "tool"

    tool = raw_tool.strip()
    if not tool:
        return "tool"

    if "<tool_call>" in tool:
        tool = tool.split("<tool_call>", 1)[0]
    if tool.endswith(")") and "(" in tool:
        inner = tool.rsplit("(", 1)[1][:-1].strip()
        if inner:
            tool = inner
    if tool.endswith("_args"):
        tool = tool[: -len("_args")]
    if tool.startswith("_"):
        tool = tool[1:]

    return _bounded_text(tool or "tool", TOOL_NAME_MAX_LENGTH)


def _looks_like_visualization_args(args: Any) -> bool:
    if not isinstance(args, dict):
        return False

    spec = args.get("spec")
    return (
        isinstance(spec, dict)
        and spec.get("page_type") in VISUALIZATION_PAGE_TYPES
        and isinstance(spec.get("title"), str)
        and isinstance(spec.get("charts"), list)
    )


def _parse_tool_result(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None

    return parsed if isinstance(parsed, dict) else None


def _extract_html_from_tool_result(value: Any) -> str | None:
    parsed = _parse_tool_result(value)
    if parsed is None:
        return None

    html = parsed.get("html")
    return html if isinstance(html, str) and html else None


def _is_visualization_event(event: dict[str, Any], tool_name: str) -> bool:
    return (
        tool_name == VISUALIZATION_TOOL_NAME
        or _looks_like_visualization_args(event.get("args"))
        or _extract_html_from_tool_result(event.get("output")) is not None
        or _extract_html_from_tool_result(event.get("output_preview")) is not None
    )


def _tool_name(event: dict[str, Any]) -> str:
    tool_name = _normalize_tool_name(event.get("tool"))
    if tool_name == "tool" and _is_visualization_event(event, tool_name):
        return VISUALIZATION_TOOL_NAME
    return tool_name


def _tool_output_preview(event: dict[str, Any], tool_name: str) -> str:
    preview = event.get("output_preview")
    is_complete = event.get("status") == "complete" or event.get("type") == "tool_end"
    if is_complete and isinstance(preview, str) and preview:
        return _bounded_text(preview, OUTPUT_PREVIEW_MAX_LENGTH)
    status = "end" if is_complete else "start"
    return _bounded_text(f"tool_{status}:{tool_name}", OUTPUT_PREVIEW_MAX_LENGTH)


def _tool_viz_html(event: dict[str, Any], tool_name: str) -> str | None:
    if event.get("status") != "complete" and event.get("type") != "tool_end":
        return None

    viz_html = event.get("viz_html")
    if isinstance(viz_html, str) and viz_html:
        return viz_html

    if not _is_visualization_event(event, tool_name):
        return None

    return _extract_html_from_tool_result(event.get("output")) or _extract_html_from_tool_result(
        event.get("output_preview")
    )


def _usage_with_ttft(
    usage: dict[str, Any] | None,
    ttft_ms: int | None,
) -> dict[str, Any] | None:
    if ttft_ms is None:
        return usage

    next_usage = dict(usage or {})
    perf = next_usage.get("perf")
    next_perf = dict(perf) if isinstance(perf, dict) else {}
    next_perf.setdefault("ttft_ms", ttft_ms)
    next_usage["perf"] = next_perf
    return next_usage


def _append_timeline_event(timeline: list[dict[str, Any]], event: dict[str, Any]) -> None:
    event_type = event.get("type")
    if event_type == "text_delta":
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        if timeline and timeline[-1].get("type") == "text":
            timeline[-1]["content"] = f"{timeline[-1].get('content', '')}{delta}"
            return
        timeline.append({"type": "text", "content": delta})
        return

    if event_type == "thinking_delta":
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        kind = event.get("kind") if isinstance(event.get("kind"), str) else "reasoning"
        if timeline and timeline[-1].get("type") == "thinking" and timeline[-1].get("kind") == kind:
            timeline[-1]["content"] = f"{timeline[-1].get('content', '')}{delta}"
            timeline[-1]["thinking"] = {kind: timeline[-1]["content"]}
            return
        timeline.append(
            {
                "type": "thinking",
                "kind": kind,
                "content": delta,
                "thinking": {kind: delta},
            }
        )
        return

    if event_type == "skill_selected":
        skill = event.get("skill")
        if isinstance(skill, str) and skill:
            timeline.append({"type": "skill", "content": skill})
        return

    if event_type == "error":
        error = event.get("error")
        if isinstance(error, str) and error:
            timeline.append({"type": "error", "content": error})
        return

    if event_type == "tool_start":
        timeline.append(
            {
                "type": "tool",
                "status": "running",
                "tool": _tool_name(event),
                "call_id": event.get("call_id"),
                "args": event.get("args"),
            }
        )
        return

    if event_type == "tool_end":
        matching_tool = _matching_tool_event(timeline, event)
        if matching_tool is None:
            matching_tool = {
                "type": "tool",
                "status": "complete",
                "tool": _tool_name(event),
                "call_id": event.get("call_id"),
            }
            timeline.append(matching_tool)
        matching_tool.update(
            {
                "status": "complete",
                "output_preview": event.get("output_preview"),
                "viz_html": event.get("viz_html"),
                "output": event.get("output"),
            }
        )


def _matching_tool_event(
    timeline: list[dict[str, Any]],
    event: dict[str, Any],
) -> dict[str, Any] | None:
    call_id = event.get("call_id")
    tool_name = _tool_name(event)
    for previous in reversed(timeline):
        if previous.get("type") != "tool" or previous.get("status") != "running":
            continue
        if call_id is not None and previous.get("call_id") == call_id:
            return previous
        if (
            call_id is None
            and previous.get("call_id") is None
            and previous.get("tool") == tool_name
        ):
            return previous
    return None


def _timeline_message(
    event: dict[str, Any],
    *,
    turn_id: str,
    transcript_sequence: int,
) -> CapturedMessage:
    common = {
        "turn_id": turn_id,
        "transcript_sequence": transcript_sequence,
    }
    if event.get("type") == "text":
        return CapturedMessage(
            role="assistant_part",
            content=str(event.get("content") or ""),
            **common,
        )

    if event.get("type") == "thinking":
        return CapturedMessage(
            role="thinking",
            content=str(event.get("content") or ""),
            thinking_json=event.get("thinking"),
            **common,
        )

    if event.get("type") in {"skill", "error"}:
        return CapturedMessage(
            role=str(event["type"]),
            content=str(event.get("content") or ""),
            **common,
        )

    tool_name = _tool_name(event)
    is_complete = event.get("status") == "complete"
    return CapturedMessage(
        role="tool",
        content=f"[{'finished' if is_complete else 'calling'} {tool_name}]",
        tool_name=tool_name,
        tool_call_id=event.get("call_id"),
        tool_input=event.get("args"),
        output_preview=_tool_output_preview(event, tool_name),
        viz_html=_tool_viz_html(event, tool_name),
        **common,
    )


def _thinking_kind(provider: str | None) -> str:
    return "summary" if provider == "openai" else "reasoning"


def _thinking_text(thinking: Any, provider: str | None) -> str:
    if isinstance(thinking, str):
        return thinking
    if not isinstance(thinking, dict):
        return ""

    preferred = _thinking_kind(provider)
    value = thinking.get(preferred)
    if isinstance(value, str) and value:
        return value

    fallback = thinking.get("reasoning") or thinking.get("summary")
    if isinstance(fallback, str):
        return fallback
    return ""
