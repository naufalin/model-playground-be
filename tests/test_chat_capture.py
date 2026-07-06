from __future__ import annotations

import json

from playground.sessions.chat_capture import (
    ChatThreadCapture,
    ChatThreadInfo,
    _normalize_tool_name,
    _tool_name,
    _tool_output_preview,
    _tool_viz_html,
)

VIZ_HTML = "<!DOCTYPE html><html><body><div id='chart'></div></body></html>"


def visualization_args() -> dict:
    return {
        "spec": {
            "page_type": "chart",
            "title": "Chart",
            "subtitle": "",
            "insights": [],
            "metrics": [],
            "charts": [
                {
                    "title": "Chart",
                    "subtitle": "",
                    "echarts_option_json": '{"series": [{"data": [1], "type": "bar"}]}',
                }
            ],
            "tables": [],
        }
    }


def capture(clock_values: list[float] | None = None) -> ChatThreadCapture:
    values = iter(clock_values or [0, 0, 0, 0])
    return ChatThreadCapture(
        ChatThreadInfo(
            id=1,
            encoded_id="thread-1",
            provider="openai",
            model_name="gpt-test",
            request_options={"provider": "openai", "model": "gpt-test"},
        ),
        monotonic=lambda: next(values),
    )


def test_capture_accumulates_text_and_prefers_done_content() -> None:
    thread = capture([0, 0.1, 0.2, 0.3])

    assert thread.observe({"type": "text_delta", "delta": "hello "}) is not None
    assert thread.observe({"type": "text_delta", "delta": "world"}) is not None
    assert (
        thread.observe({"type": "done", "content": "final", "usage": {"total_tokens": 3}})
        is None
    )

    done = thread.thread_done_event()
    messages = thread.captured_messages()

    assert done["content"] == "final"
    assert done["usage"]["perf"]["ttft_ms"] == 100
    assert messages[-1].content == "final"
    assert messages[-1].usage_json["perf"]["ttft_ms"] == 100


def test_capture_adds_ttft_when_runtime_usage_omits_it() -> None:
    thread = capture([0, 0.25, 0.5])

    thread.observe({"type": "text_delta", "delta": "hello"})
    thread.observe({"type": "done", "usage": {"output_tokens": 2}})

    assert thread.thread_done_event()["usage"] == {
        "output_tokens": 2,
        "perf": {"ttft_ms": 250},
    }


def test_capture_merges_thinking_delta_by_kind() -> None:
    thread = capture()

    thread.observe({"type": "thinking_delta", "kind": "reasoning", "delta": "one "})
    thread.observe({"type": "thinking_delta", "kind": "reasoning", "delta": "two"})

    messages = thread.captured_messages()

    assert len(messages) == 1
    assert messages[0].role == "thinking"
    assert messages[0].content == "one two"
    assert messages[0].thinking_json == {"reasoning": "one two"}


def test_capture_uses_done_only_thinking_fallback() -> None:
    thread = capture()

    thread.observe(
        {
            "type": "done",
            "content": "answer",
            "provider": "openai",
            "thinking": {"summary": "final summary"},
        }
    )

    messages = thread.captured_messages()

    assert [message.role for message in messages] == ["thinking", "assistant"]
    assert messages[0].content == "final summary"
    assert messages[0].thinking_json == {"summary": "final summary"}


def test_capture_copies_tool_start_args_to_matching_end() -> None:
    thread = capture()

    thread.observe(
        {
            "type": "tool_start",
            "tool": "web_search",
            "call_id": "call-1",
            "args": {"query": "hello"},
        }
    )
    thread.observe(
        {
            "type": "tool_end",
            "tool": "web_search",
            "call_id": "call-1",
            "output_preview": "done",
        }
    )

    messages = thread.captured_messages()

    assert messages[0].tool_input == {"query": "hello"}
    assert messages[1].tool_input is None
    assert messages[1].output_preview == "done"


def test_capture_persists_visualization_html_from_all_runtime_shapes() -> None:
    direct = capture()
    output = capture()
    preview = capture()

    direct.observe(
        {
            "type": "tool_end",
            "tool": "generate_visualization",
            "call_id": "viz-1",
            "viz_html": VIZ_HTML,
        }
    )
    output.observe(
        {
            "type": "tool_end",
            "tool": "generate_visualization",
            "call_id": "viz-2",
            "output": {"html": VIZ_HTML},
        }
    )
    preview.observe(
        {
            "type": "tool_end",
            "tool": "generate_visualization",
            "call_id": "viz-3",
            "output_preview": json.dumps({"html": VIZ_HTML}),
        }
    )

    assert direct.captured_messages()[0].viz_html == VIZ_HTML
    assert output.captured_messages()[0].viz_html == VIZ_HTML
    assert preview.captured_messages()[0].viz_html == VIZ_HTML


def test_capture_error_does_not_create_assistant_without_content() -> None:
    thread = capture([0, 0.1])

    assert thread.error_event(RuntimeError("runtime failed")) == {
        "type": "error",
        "thread_id": "thread-1",
        "error": "runtime failed",
    }

    assert thread.captured_messages() == []


def test_normalize_tool_name_extracts_markup_tool_label() -> None:
    assert (
        _normalize_tool_name(
            "web_search_args(_web_search)<tool_call>query</arg_key>"
            "<arg_value>Cut Nyak Dien</arg_value>"
        )
        == "web_search"
    )


def test_tool_name_infers_visualization_from_generic_spec_args() -> None:
    assert (
        _tool_name(
            {
                "type": "tool_start",
                "tool": "tool",
                "args": visualization_args(),
            }
        )
        == "generate_visualization"
    )


def test_tool_output_preview_falls_back_for_old_runtime_events() -> None:
    assert (
        _tool_output_preview(
            {"type": "tool_end", "tool": "web_search", "call_id": "call-1"},
            "web_search",
        )
        == "tool_end:web_search"
    )


def test_tool_output_preview_truncates_long_values() -> None:
    preview = _tool_output_preview({"type": "tool_end", "output_preview": "x" * 600}, "tool")

    assert len(preview) == 500


def test_tool_viz_html_falls_back_to_output_json() -> None:
    assert (
        _tool_viz_html(
            {
                "type": "tool_end",
                "tool": "generate_visualization",
                "output": {"html": VIZ_HTML, "title": "Chart"},
            },
            "generate_visualization",
        )
        == VIZ_HTML
    )


def test_tool_viz_html_falls_back_to_output_preview_json() -> None:
    assert (
        _tool_viz_html(
            {
                "type": "tool_end",
                "tool": "generate_visualization",
                "output_preview": json.dumps({"html": VIZ_HTML, "title": "Chart"}),
            },
            "generate_visualization",
        )
        == VIZ_HTML
    )
