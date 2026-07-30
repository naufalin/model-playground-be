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


def test_capture_forwards_and_persists_selected_skill() -> None:
    thread = capture()

    selected = thread.observe({"type": "skill_selected", "skill": "debugger"})
    thread.observe({"type": "done", "content": "fixed", "selected_skill": "debugger"})

    assert selected == {"type": "skill_selected", "skill": "debugger"}
    assert thread.thread_done_event()["selected_skill"] == "debugger"
    assert thread.captured_messages()[-1].selected_skill == "debugger"


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

    assert [message.role for message in messages] == [
        "thinking",
        "assistant_part",
        "assistant",
    ]
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

    assert len(messages) == 1
    assert messages[0].tool_input == {"query": "hello"}
    assert messages[0].output_preview == "done"


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


def test_capture_error_creates_ordered_error_without_assistant() -> None:
    thread = capture([0, 0.1])

    assert thread.error_event(RuntimeError("runtime failed")) == {
        "type": "error",
        "thread_id": "thread-1",
        "error": "runtime failed",
    }

    messages = thread.captured_messages()
    assert [(message.role, message.content) for message in messages] == [
        ("error", "runtime failed")
    ]


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


def test_capture_preserves_text_thinking_text_order_without_inserting_spaces() -> None:
    thread = capture()

    thread.observe({"type": "text_delta", "delta": "you."})
    thread.observe({"type": "thinking_delta", "kind": "reasoning", "delta": "consider"})
    thread.observe({"type": "text_delta", "delta": "I found"})
    thread.observe({"type": "done", "content": "you.I found"})

    messages = thread.captured_messages()

    assert [message.role for message in messages] == [
        "assistant_part",
        "thinking",
        "assistant_part",
        "assistant",
    ]
    assert [message.content for message in messages] == [
        "you.",
        "consider",
        "I found",
        "you.I found",
    ]
    assert [message.transcript_sequence for message in messages] == [0, 1, 2, None]
    assert len({message.turn_id for message in messages}) == 1


def test_capture_keeps_tool_at_start_position_and_updates_completion() -> None:
    thread = capture()

    thread.observe({"type": "text_delta", "delta": "before"})
    thread.observe({"type": "skill_selected", "skill": "researcher"})
    thread.observe(
        {
            "type": "tool_start",
            "tool": "web_search",
            "call_id": "call-1",
            "args": {"query": "nasi goreng"},
        }
    )
    thread.observe({"type": "thinking_delta", "kind": "reasoning", "delta": "checking"})
    thread.observe(
        {
            "type": "tool_end",
            "tool": "web_search",
            "call_id": "call-1",
            "output_preview": "three results",
        }
    )
    thread.observe({"type": "text_delta", "delta": "after"})
    thread.observe({"type": "done", "content": "beforeafter"})

    messages = thread.captured_messages()

    assert [message.role for message in messages] == [
        "assistant_part",
        "skill",
        "tool",
        "thinking",
        "assistant_part",
        "assistant",
    ]
    assert messages[2].tool_input == {"query": "nasi goreng"}
    assert messages[2].output_preview == "three results"


def test_capture_appends_done_suffix_after_intervening_activity() -> None:
    thread = capture()

    thread.observe({"type": "text_delta", "delta": "first"})
    thread.observe({"type": "thinking_delta", "kind": "summary", "delta": "pause"})
    thread.observe({"type": "done", "content": "first suffix"})

    messages = thread.captured_messages()

    assert [(message.role, message.content) for message in messages] == [
        ("assistant_part", "first"),
        ("thinking", "pause"),
        ("assistant_part", " suffix"),
        ("assistant", "first suffix"),
    ]


def test_capture_drops_text_fragments_when_done_replaces_streamed_text() -> None:
    thread = capture()

    thread.observe({"type": "text_delta", "delta": "draft"})
    thread.observe({"type": "thinking_delta", "kind": "summary", "delta": "pause"})
    thread.observe({"type": "done", "content": "corrected"})

    messages = thread.captured_messages()

    assert [(message.role, message.content) for message in messages] == [
        ("thinking", "pause"),
        ("assistant", "corrected"),
    ]


def test_capture_supports_overlapping_and_unmatched_tool_calls() -> None:
    thread = capture()

    thread.observe(
        {"type": "tool_start", "tool": "first", "call_id": "call-1", "args": {"n": 1}}
    )
    thread.observe(
        {"type": "tool_start", "tool": "second", "call_id": "call-2", "args": {"n": 2}}
    )
    thread.observe(
        {
            "type": "tool_end",
            "tool": "second",
            "call_id": "call-2",
            "output_preview": "second done",
        }
    )
    thread.observe(
        {
            "type": "tool_end",
            "tool": "missing",
            "call_id": "call-3",
            "output_preview": "unmatched done",
        }
    )
    thread.observe(
        {
            "type": "tool_end",
            "tool": "first",
            "call_id": "call-1",
            "output_preview": "first done",
        }
    )

    messages = thread.captured_messages()

    assert [message.tool_call_id for message in messages] == ["call-1", "call-2", "call-3"]
    assert [message.output_preview for message in messages] == [
        "first done",
        "second done",
        "unmatched done",
    ]


def test_capture_honors_empty_done_content_as_canonical_replacement() -> None:
    thread = capture()

    thread.observe({"type": "text_delta", "delta": "draft"})
    thread.observe({"type": "done", "content": ""})

    messages = thread.captured_messages()

    assert [(message.role, message.content) for message in messages] == [("assistant", "")]
