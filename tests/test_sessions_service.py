from __future__ import annotations

import json

import pytest

from playground.db.connection import Database
from playground.db.models import Base, LlmModel, ModelThread, PlaygroundSession, User
from playground.db.repos.thread_repo import ThreadRepo
from playground.ids import encode
from playground.sessions.service import (
    ModelNotFoundError,
    PlaygroundNotFoundError,
    PlaygroundService,
    _normalize_tool_name,
    _tool_output_preview,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.created_tools: list[list[str] | None] = []
        self.chat_tools: list[list[str] | None] = []

    async def create_session(
        self,
        title: str = "New Session",
        tools: list[str] | None = None,
    ) -> str:
        self.created.append(title)
        self.created_tools.append(tools)
        return f"runtime-{title}"

    async def chat_stream(
        self,
        session_id: str,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        tools: list[str] | None = None,
    ):
        self.chat_tools.append(tools)
        yield {
            "type": "thinking_delta",
            "delta": "thinking",
            "kind": "reasoning",
        }
        yield {
            "type": "tool_start",
            "tool": "web_search",
            "call_id": "call-1",
            "args": {"query": "hello"},
        }
        yield {
            "type": "tool_end",
            "tool": "web_search",
            "call_id": "call-1",
            "output_preview": '{"results":[{"title":"Gold price"}]}',
        }
        yield {"type": "text_delta", "delta": "hello "}
        yield {"type": "text_delta", "delta": "world"}
        yield {
            "type": "done",
            "provider": provider,
            "model": model,
            "usage": {"total_tokens": 10, "reasoning_tokens": 2, "perf": {"ttft_ms": 5}},
            "thinking": {"reasoning": "visible thought"},
            "output_delta_count": 2,
        }


class ErrorRuntime(FakeRuntime):
    async def chat_stream(self, session_id: str, message: str, **kwargs):
        raise RuntimeError("runtime failed")
        yield


class DoneOnlyThinkingRuntime(FakeRuntime):
    async def chat_stream(
        self,
        session_id: str,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        tools: list[str] | None = None,
    ):
        self.chat_tools.append(tools)
        yield {"type": "text_delta", "delta": "done-only"}
        yield {
            "type": "done",
            "provider": provider,
            "model": model,
            "usage": {"total_tokens": 4},
            "thinking": {"summary": "final summary"},
            "output_delta_count": 1,
        }


class VisualizationRuntime(FakeRuntime):
    async def chat_stream(
        self,
        session_id: str,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        tools: list[str] | None = None,
    ):
        self.chat_tools.append(tools)
        yield {
            "type": "tool_start",
            "tool": "tool",
            "call_id": "viz-1",
            "args": {"chart_type": "bar"},
        }
        yield {
            "type": "tool_end",
            "tool": "tool",
            "call_id": "viz-1",
            "output_preview": '{"html":"<!DOCTYPE html>"}',
            "viz_html": "<!DOCTYPE html><html><body><div id='chart'></div></body></html>",
        }
        yield {"type": "text_delta", "delta": "Here is the chart."}
        yield {
            "type": "done",
            "provider": provider,
            "model": model,
            "usage": {"total_tokens": 7},
            "thinking": None,
            "output_delta_count": 1,
        }


class MissingTtftRuntime(FakeRuntime):
    async def chat_stream(
        self,
        session_id: str,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        tools: list[str] | None = None,
    ):
        self.chat_tools.append(tools)
        yield {"type": "text_delta", "delta": "hello"}
        yield {
            "type": "done",
            "provider": provider,
            "model": model,
            "usage": {"output_tokens": 2, "total_tokens": 3},
            "thinking": None,
            "output_delta_count": 1,
        }


class MarkupToolRuntime(FakeRuntime):
    async def chat_stream(
        self,
        session_id: str,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        tools: list[str] | None = None,
    ):
        self.chat_tools.append(tools)
        yield {
            "type": "tool_start",
            "tool": (
                "web_search_args(_web_search)<tool_call>query</arg_key>"
                "<arg_value>Cut Nyak Dien pahlawan perjuangan Aceh Belanda</arg_value>"
            ),
            "call_id": "call-1",
            "args": {"location": "ID", "language": "id", "page": "0"},
        }
        yield {"type": "text_delta", "delta": "done"}
        yield {
            "type": "done",
            "provider": provider,
            "model": model,
            "usage": {"total_tokens": 3},
            "thinking": None,
            "output_delta_count": 1,
        }


@pytest.fixture
async def db() -> Database:
    database = Database("sqlite+aiosqlite:///:memory:")
    database.connect()
    assert database.engine is not None
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield database
    await database.disconnect()


async def create_user(db: Database, email: str = "user@example.com") -> User:
    async with db.session() as session:
        user = User(email=email, hashed_password="hashed")
        session.add(user)
        await session.flush()
        return user


async def create_model(db: Database) -> LlmModel:
    async with db.session() as session:
        model = LlmModel(
            provider="openai",
            model_name="gpt-test",
            display_name="GPT Test",
            is_active=True,
        )
        session.add(model)
        await session.flush()
        return model


async def create_session(db: Database, user_id: int) -> PlaygroundSession:
    async with db.session() as session:
        playground = PlaygroundSession(user_id=user_id, title="Existing")
        session.add(playground)
        await session.flush()
        return playground


async def test_service_creates_and_lists_playgrounds_with_total(db: Database) -> None:
    user = await create_user(db)
    service = PlaygroundService(db, FakeRuntime())

    created = await service.create_playground(user.id, "Side by side")
    listed = await service.list_playgrounds(user.id, limit=20, offset=0)

    assert created.title == "Side by side"
    assert listed.total == 1
    assert listed.sessions[0].id == created.id


async def test_service_rejects_playground_owned_by_another_user(db: Database) -> None:
    owner = await create_user(db, "owner@example.com")
    other = await create_user(db, "other@example.com")
    playground = await create_session(db, owner.id)
    service = PlaygroundService(db, FakeRuntime())

    with pytest.raises(PlaygroundNotFoundError):
        await service.get_playground(encode(playground.id), other.id)


async def test_service_updates_playground_title_for_owner(db: Database) -> None:
    user = await create_user(db)
    playground = await create_session(db, user.id)
    service = PlaygroundService(db, FakeRuntime())

    updated = await service.update_playground(encode(playground.id), user.id, "Renamed")
    detail = await service.get_playground(encode(playground.id), user.id)

    assert updated.title == "Renamed"
    assert detail.title == "Renamed"


async def test_service_rejects_title_update_for_another_user(db: Database) -> None:
    owner = await create_user(db, "owner@example.com")
    other = await create_user(db, "other@example.com")
    playground = await create_session(db, owner.id)
    service = PlaygroundService(db, FakeRuntime())

    with pytest.raises(PlaygroundNotFoundError):
        await service.update_playground(encode(playground.id), other.id, "Nope")


async def test_multi_chat_creates_threads_only_for_valid_models(db: Database) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    runtime = FakeRuntime()
    service = PlaygroundService(db, runtime)

    await service.stream_multi_chat(
        encode(playground.id),
        user.id,
        "hello",
        [(model.provider, model.model_name, None)],
        ["web_search"],
    )

    async with db.session() as session:
        threads = await ThreadRepo(session).get_by_session(playground.id)

    assert runtime.created == ["openai/gpt-test"]
    assert runtime.created_tools == [["web_search"]]
    assert len(threads) == 1
    assert threads[0].runtime_session_id == "runtime-openai/gpt-test"

    with pytest.raises(ModelNotFoundError):
        await service.stream_multi_chat(
            encode(playground.id),
            user.id,
            "hello",
            [("openai", "missing", None)],
        )


async def test_multi_chat_rejects_inactive_models(db: Database) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    async with db.session() as session:
        stored = await session.get(LlmModel, model.id)
        assert stored is not None
        stored.is_active = False

    service = PlaygroundService(db, FakeRuntime())

    with pytest.raises(ModelNotFoundError):
        await service.stream_multi_chat(
            encode(playground.id),
            user.id,
            "hello",
            [(model.provider, model.model_name, None)],
        )


async def test_multi_chat_persists_user_and_assistant_messages(db: Database) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    service = PlaygroundService(db, FakeRuntime())

    stream = await service.stream_multi_chat(
        encode(playground.id),
        user.id,
        "hello",
        [(model.provider, model.model_name, "high")],
        ["web_search"],
    )
    chunks = [chunk async for chunk in stream]

    async with db.session() as session:
        threads = await ThreadRepo(session).get_by_session(playground.id)
        messages = threads[0].messages

    assert chunks[-1] == 'data: {"type": "all_done"}\n\n'
    assert [message.role for message in messages] == [
        "user",
        "thinking",
        "tool",
        "tool",
        "assistant",
    ]
    assert messages[0].content == "hello"
    assert messages[0].request_options_json == {
        "provider": "openai",
        "model": "gpt-test",
        "reasoning_effort": "high",
    }
    assert messages[1].content == "thinking"
    assert messages[1].thinking_json == {"reasoning": "thinking"}
    assert messages[2].tool_name == "web_search"
    assert messages[2].tool_input == {"query": "hello"}
    assert messages[3].output_preview == '{"results":[{"title":"Gold price"}]}'
    assert messages[4].content == "hello world"
    assert messages[4].provider == "openai"
    assert messages[4].model == "gpt-test"
    assert messages[4].usage_json["reasoning_tokens"] == 2
    assert messages[4].thinking_json["reasoning"] == "visible thought"
    assert messages[4].output_delta_count == 2
    assert service.runtime.chat_tools == [["web_search"]]


async def test_multi_chat_persists_messages_before_all_done(db: Database) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    service = PlaygroundService(db, FakeRuntime())

    stream = await service.stream_multi_chat(
        encode(playground.id),
        user.id,
        "hello",
        [(model.provider, model.model_name, None)],
    )

    saw_all_done = False
    async for chunk in stream:
        event = json.loads(chunk.removeprefix("data: ").strip())
        if event.get("type") != "all_done":
            continue

        saw_all_done = True
        async with db.session() as session:
            threads = await ThreadRepo(session).get_by_session(playground.id)
            messages = threads[0].messages

        assert [message.role for message in messages] == [
            "user",
            "thinking",
            "tool",
            "tool",
            "assistant",
        ]
        assert messages[-1].content == "hello world"

    assert saw_all_done


async def test_multi_chat_persists_markup_tool_name_without_overflow(db: Database) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    service = PlaygroundService(db, MarkupToolRuntime())

    stream = await service.stream_multi_chat(
        encode(playground.id),
        user.id,
        "hello",
        [(model.provider, model.model_name, None)],
    )
    chunks = [chunk async for chunk in stream]

    async with db.session() as session:
        threads = await ThreadRepo(session).get_by_session(playground.id)
        messages = threads[0].messages

    assert chunks[-1] == 'data: {"type": "all_done"}\n\n'
    tool_message = [message for message in messages if message.role == "tool"][0]
    assert tool_message.tool_name == "web_search"
    assert len(tool_message.tool_name) <= 100
    assert tool_message.content == "[calling web_search]"
    assert "<tool_call>" not in tool_message.content
    assert tool_message.tool_input == {"location": "ID", "language": "id", "page": "0"}


async def test_multi_chat_persists_done_only_thinking_before_assistant(
    db: Database,
) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    service = PlaygroundService(db, DoneOnlyThinkingRuntime())

    stream = await service.stream_multi_chat(
        encode(playground.id),
        user.id,
        "hello",
        [(model.provider, model.model_name, "high")],
    )
    chunks = [chunk async for chunk in stream]

    async with db.session() as session:
        threads = await ThreadRepo(session).get_by_session(playground.id)
        messages = threads[0].messages

    assert chunks[-1] == 'data: {"type": "all_done"}\n\n'
    assert [message.role for message in messages] == ["user", "thinking", "assistant"]
    assert messages[1].content == "final summary"
    assert messages[1].thinking_json == {"summary": "final summary"}
    assert messages[2].content == "done-only"


async def test_multi_chat_persists_visualization_html(db: Database) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    service = PlaygroundService(db, VisualizationRuntime())

    stream = await service.stream_multi_chat(
        encode(playground.id),
        user.id,
        "make a chart",
        [(model.provider, model.model_name, None)],
    )
    chunks = [chunk async for chunk in stream]

    async with db.session() as session:
        threads = await ThreadRepo(session).get_by_session(playground.id)
        tool_messages = [message for message in threads[0].messages if message.role == "tool"]

    detail = await service.get_playground(encode(playground.id), user.id)
    detail_tool_messages = [
        message for message in detail.threads[0].messages if message.role == "tool"
    ]

    assert '"viz_html": "<!DOCTYPE html>' in "".join(chunks)
    assert tool_messages[-1].tool_name == "tool"
    assert tool_messages[-1].viz_html == (
        "<!DOCTYPE html><html><body><div id='chart'></div></body></html>"
    )
    assert detail_tool_messages[-1].viz_html == tool_messages[-1].viz_html


async def test_single_chat_adds_ttft_when_runtime_usage_omits_it(db: Database) -> None:
    user = await create_user(db)
    playground = await create_session(db, user.id)
    async with db.session() as session:
        thread = ModelThread(
            playground_session_id=playground.id,
            provider="openai",
            model_name="gpt-test",
            runtime_session_id="runtime-existing",
        )
        session.add(thread)
        await session.flush()
        thread_id = thread.id

    service = PlaygroundService(db, MissingTtftRuntime())
    stream = await service.stream_single_chat(
        encode(playground.id),
        encode(thread_id),
        user.id,
        "hello",
        ["web_search"],
    )
    chunks = [chunk async for chunk in stream]

    async with db.session() as session:
        stored = await ThreadRepo(session).get(thread_id)
        assert stored is not None
        assistant = [message for message in stored.messages if message.role == "assistant"][0]

    assert any('"type": "thread_start"' in chunk for chunk in chunks)
    assert assistant.usage_json["perf"]["ttft_ms"] >= 0
    assert '"perf": {"ttft_ms":' in "".join(chunks)
    assert service.runtime.chat_tools == [["web_search"]]


def test_normalize_tool_name_extracts_markup_tool_label() -> None:
    assert (
        _normalize_tool_name(
            "web_search_args(_web_search)<tool_call>query</arg_key>"
            "<arg_value>Cut Nyak Dien</arg_value>"
        )
        == "web_search"
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


async def test_single_chat_emits_error_event_when_runtime_fails(db: Database) -> None:
    user = await create_user(db)
    playground = await create_session(db, user.id)
    async with db.session() as session:
        thread = ModelThread(
            playground_session_id=playground.id,
            provider="openai",
            model_name="gpt-test",
            runtime_session_id="runtime-existing",
        )
        session.add(thread)
        await session.flush()
        thread_id = thread.id

    service = PlaygroundService(db, ErrorRuntime())
    stream = await service.stream_single_chat(
        encode(playground.id),
        encode(thread_id),
        user.id,
        "hello",
    )
    chunks = [chunk async for chunk in stream]

    assert any('"type": "error"' in chunk and "runtime failed" in chunk for chunk in chunks)
    assert chunks[-1] == 'data: {"type": "all_done"}\n\n'
