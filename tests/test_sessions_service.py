from __future__ import annotations

import asyncio
import json

import pytest

from playground.db.connection import Database
from playground.db.models import Base, LlmModel, ModelThread, PlaygroundSession, User
from playground.db.repos.thread_repo import ThreadRepo
from playground.ids import encode
from playground.sessions.service import (
    MessageNotFoundError,
    ModelNotFoundError,
    PlaygroundNotFoundError,
    PlaygroundService,
    RuntimeForkError,
    SystemPromptLockedError,
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


class FakeRuntime:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.created_tools: list[list[str] | None] = []
        self.created_skills: list[list[str] | None] = []
        self.created_system_prompts: list[str | None] = []
        self.chat_tools: list[list[str] | None] = []
        self.chat_skills: list[list[str] | None] = []
        self.chat_session_ids: list[str] = []
        self.chat_reasoning_efforts: list[str | None] = []
        self.fork_calls: list[tuple[str, int]] = []

    async def list_prompts(self) -> dict:
        return {
            "prompts": [
                {
                    "id": 1,
                    "name": "default",
                    "content": "You are a helpful assistant.",
                }
            ]
        }

    async def create_session(
        self,
        title: str = "New Session",
        tools: list[str] | None = None,
        skills: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        self.created.append(title)
        self.created_tools.append(tools)
        self.created_skills.append(skills)
        self.created_system_prompts.append(system_prompt)
        return f"runtime-{title}"

    async def fork_session(self, session_id: str, keep_user_turns: int) -> str:
        self.fork_calls.append((session_id, keep_user_turns))
        return f"fork-{session_id}-{keep_user_turns}"

    async def chat_stream(
        self,
        session_id: str,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        tools: list[str] | None = None,
        skills: list[str] | None = None,
    ):
        self.chat_tools.append(tools)
        self.chat_skills.append(skills)
        self.chat_session_ids.append(session_id)
        self.chat_reasoning_efforts.append(reasoning_effort)
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


class PartialBlockingRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = asyncio.Event()

    async def chat_stream(self, session_id: str, message: str, **kwargs):
        try:
            yield {"type": "text_delta", "delta": "partial answer"}
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()


class ForkErrorRuntime(FakeRuntime):
    async def fork_session(self, session_id: str, keep_user_turns: int) -> str:
        raise RuntimeError("fork failed")


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
        skills: list[str] | None = None,
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
        skills: list[str] | None = None,
    ):
        self.chat_tools.append(tools)
        yield {
            "type": "tool_start",
            "tool": "tool",
            "call_id": "viz-1",
            "args": visualization_args(),
        }
        yield {
            "type": "tool_end",
            "tool": "tool",
            "call_id": "viz-1",
            "output_preview": '{"html":"<!DOCTYPE html>"}',
            "viz_html": VIZ_HTML,
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


class VisualizationOutputFallbackRuntime(FakeRuntime):
    async def chat_stream(
        self,
        session_id: str,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        tools: list[str] | None = None,
        skills: list[str] | None = None,
    ):
        self.chat_tools.append(tools)
        yield {
            "type": "tool_start",
            "tool": "tool",
            "call_id": "viz-1",
            "args": visualization_args(),
        }
        yield {
            "type": "tool_end",
            "tool": "tool",
            "call_id": "viz-1",
            "output": {"html": VIZ_HTML, "title": "Chart"},
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
        skills: list[str] | None = None,
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
        skills: list[str] | None = None,
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
    assert created.system_prompt_name == "Default"
    assert created.system_prompt_content == "You are a helpful assistant."
    assert listed.total == 1
    assert listed.sessions[0].id == created.id
    assert listed.sessions[0].system_prompt_content is None


async def test_service_persists_system_prompt_snapshot(db: Database) -> None:
    user = await create_user(db)
    service = PlaygroundService(db, FakeRuntime())

    created = await service.create_playground(
        user.id,
        "Prompt comparison",
        system_prompt_name="Concise analyst",
        system_prompt_content="You are a concise analyst.",
    )
    detail = await service.get_playground(created.id, user.id)

    assert created.system_prompt_name == "Concise analyst"
    assert created.system_prompt_content == "You are a concise analyst."
    assert detail.system_prompt_name == "Concise analyst"
    assert detail.system_prompt_content == "You are a concise analyst."


async def test_service_preserves_session_skill_states(db: Database) -> None:
    user = await create_user(db)
    service = PlaygroundService(db, FakeRuntime())

    created = await service.create_playground(
        user.id, "Skilled", skills=["debugger"]
    )
    disabled = await service.update_playground(
        created.id, user.id, skills=[], update_skills=True
    )
    defaults = await service.update_playground(
        created.id, user.id, skills=None, update_skills=True
    )

    assert created.skills == ["debugger"]
    assert disabled.skills == []
    assert defaults.skills is None


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


async def test_multi_chat_applies_playground_system_prompt_to_runtime_sessions(
    db: Database,
) -> None:
    user = await create_user(db)
    model = await create_model(db)
    service = PlaygroundService(db, FakeRuntime())
    created = await service.create_playground(
        user.id,
        "Prompt comparison",
        system_prompt_name="Custom",
        system_prompt_content="Return JSON only.",
    )

    await service.stream_multi_chat(
        created.id,
        user.id,
        "hello",
        [(model.provider, model.model_name, None)],
    )

    runtime = service.runtime
    assert isinstance(runtime, FakeRuntime)
    assert runtime.created_system_prompts == ["Return JSON only."]


async def test_service_locks_system_prompt_after_comparison_starts(
    db: Database,
) -> None:
    user = await create_user(db)
    model = await create_model(db)
    service = PlaygroundService(db, FakeRuntime())
    created = await service.create_playground(
        user.id,
        "Prompt comparison",
        system_prompt_name="Default",
        system_prompt_content="You are helpful.",
    )
    await service.stream_multi_chat(
        created.id,
        user.id,
        "hello",
        [(model.provider, model.model_name, None)],
    )

    with pytest.raises(SystemPromptLockedError):
        await service.update_playground(
            created.id,
            user.id,
            system_prompt_name="Changed",
            system_prompt_content="Behave differently.",
            update_system_prompt=True,
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
    detail = await service.get_playground(encode(playground.id), user.id)
    detail_messages = detail.threads[0].messages

    assert chunks[-1] == 'data: {"type": "all_done"}\n\n'
    assert [message.role for message in messages] == [
        "user",
        "thinking",
        "tool",
        "assistant_part",
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
    assert messages[2].output_preview == '{"results":[{"title":"Gold price"}]}'
    assert messages[3].content == "hello world"
    assert messages[4].content == "hello world"
    assert messages[4].provider == "openai"
    assert messages[4].model == "gpt-test"
    assert messages[4].usage_json["reasoning_tokens"] == 2
    assert messages[4].thinking_json["reasoning"] == "visible thought"
    assert messages[4].output_delta_count == 2
    assert [message.transcript_sequence for message in messages[1:]] == [0, 1, 2, None]
    assert [message.transcript_sequence for message in detail_messages[1:]] == [0, 1, 2, None]
    assert detail_messages[-1].turn_id == detail_messages[-2].turn_id
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
            "assistant_part",
            "assistant",
        ]
        assert messages[-1].content == "hello world"

    assert saw_all_done


async def test_closing_multi_chat_cancels_runtime_and_persists_partial_output(
    db: Database,
) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    runtime = PartialBlockingRuntime()
    service = PlaygroundService(db, runtime)
    stream = await service.stream_multi_chat(
        encode(playground.id),
        user.id,
        "hello",
        [(model.provider, model.model_name, None)],
    )

    while True:
        chunk = await anext(stream)
        if '"type": "text_delta"' in chunk:
            break
    await stream.aclose()

    async with db.session() as session:
        threads = await ThreadRepo(session).get_by_session(playground.id)
        assistant = [message for message in threads[0].messages if message.role == "assistant"][-1]

    assert runtime.cancelled.is_set()
    assert assistant.content == "partial answer"
    assert assistant.usage_json["cancelled"] is True


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
    assert [message.role for message in messages] == [
        "user",
        "thinking",
        "assistant_part",
        "assistant",
    ]
    assert messages[1].content == "final summary"
    assert messages[1].thinking_json == {"summary": "final summary"}
    assert messages[2].content == "done-only"
    assert messages[3].content == "done-only"


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
    assert [message.tool_name for message in tool_messages] == ["generate_visualization"]
    assert tool_messages[-1].viz_html == VIZ_HTML
    assert detail_tool_messages[-1].viz_html == tool_messages[-1].viz_html


async def test_multi_chat_extracts_visualization_html_from_output_fallback(
    db: Database,
) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    service = PlaygroundService(db, VisualizationOutputFallbackRuntime())

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

    assert chunks[-1] == 'data: {"type": "all_done"}\n\n'
    assert [message.tool_name for message in tool_messages] == ["generate_visualization"]
    assert tool_messages[-1].viz_html == VIZ_HTML


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


async def test_closing_single_chat_cancels_runtime_and_persists_partial_output(
    db: Database,
) -> None:
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

    runtime = PartialBlockingRuntime()
    service = PlaygroundService(db, runtime)
    stream = await service.stream_single_chat(
        encode(playground.id),
        encode(thread_id),
        user.id,
        "hello",
    )

    assert '"type": "thread_start"' in await anext(stream)
    assert '"type": "text_delta"' in await anext(stream)
    await stream.aclose()

    async with db.session() as session:
        stored = await ThreadRepo(session).get(thread_id)
        assert stored is not None
        assistant = [message for message in stored.messages if message.role == "assistant"][-1]

    assert runtime.cancelled.is_set()
    assert assistant.content == "partial answer"
    assert assistant.usage_json["cancelled"] is True


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


async def test_regenerated_chat_replaces_the_thread_tail_and_reuses_reasoning(
    db: Database,
) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    runtime = FakeRuntime()
    service = PlaygroundService(db, runtime)

    initial_stream = await service.stream_multi_chat(
        encode(playground.id),
        user.id,
        "Original prompt",
        [(model.provider, model.model_name, "high")],
        ["web_search"],
    )
    _ = [chunk async for chunk in initial_stream]

    async with db.session() as session:
        thread = (await ThreadRepo(session).get_by_session(playground.id))[0]
        thread_id = thread.id
        original_message = next(message for message in thread.messages if message.role == "user")
        original_message_id = original_message.id

    stream = await service.stream_regenerated_chat(
        encode(playground.id),
        encode(thread_id),
        original_message_id,
        user.id,
        "Edited prompt",
        [],
    )
    chunks = [chunk async for chunk in stream]

    async with db.session() as session:
        updated = await ThreadRepo(session).get(thread_id)
        assert updated is not None
        messages = updated.messages

    assert runtime.fork_calls == [("runtime-openai/gpt-test", 0)]
    assert runtime.chat_session_ids[-1] == "fork-runtime-openai/gpt-test-0"
    assert runtime.chat_tools[-1] == []
    assert runtime.chat_reasoning_efforts[-1] == "high"
    assert updated.runtime_session_id == "fork-runtime-openai/gpt-test-0"
    assert [message.role for message in messages] == [
        "user",
        "thinking",
        "tool",
        "assistant_part",
        "assistant",
    ]
    assert messages[0].content == "Edited prompt"
    assert messages[0].request_options_json == {
        "provider": "openai",
        "model": "gpt-test",
        "reasoning_effort": "high",
    }
    assert any('"type": "thread_start"' in chunk for chunk in chunks)
    assert chunks[-1] == 'data: {"type": "all_done"}\n\n'


async def test_regenerated_chat_keeps_history_when_runtime_fork_fails(db: Database) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    service = PlaygroundService(db, ForkErrorRuntime())

    initial_stream = await service.stream_multi_chat(
        encode(playground.id),
        user.id,
        "Original prompt",
        [(model.provider, model.model_name, None)],
    )
    _ = [chunk async for chunk in initial_stream]

    async with db.session() as session:
        thread = (await ThreadRepo(session).get_by_session(playground.id))[0]
        thread_id = thread.id
        message_id = next(message.id for message in thread.messages if message.role == "user")
        original_runtime_session_id = thread.runtime_session_id
        original_contents = [message.content for message in thread.messages]

    with pytest.raises(RuntimeForkError, match="Could not prepare message regeneration"):
        await service.stream_regenerated_chat(
            encode(playground.id),
            encode(thread_id),
            message_id,
            user.id,
            "Edited prompt",
        )

    async with db.session() as session:
        unchanged = await ThreadRepo(session).get(thread_id)
        assert unchanged is not None
        assert unchanged.runtime_session_id == original_runtime_session_id
        assert [message.content for message in unchanged.messages] == original_contents


async def test_regenerated_chat_rejects_assistant_messages(db: Database) -> None:
    user = await create_user(db)
    model = await create_model(db)
    playground = await create_session(db, user.id)
    runtime = FakeRuntime()
    service = PlaygroundService(db, runtime)

    initial_stream = await service.stream_multi_chat(
        encode(playground.id),
        user.id,
        "Original prompt",
        [(model.provider, model.model_name, None)],
    )
    _ = [chunk async for chunk in initial_stream]

    async with db.session() as session:
        thread = (await ThreadRepo(session).get_by_session(playground.id))[0]
        assistant_id = next(
            message.id for message in thread.messages if message.role == "assistant"
        )

    with pytest.raises(MessageNotFoundError, match="User message not found"):
        await service.stream_regenerated_chat(
            encode(playground.id),
            encode(thread.id),
            assistant_id,
            user.id,
            "Edited prompt",
        )

    assert runtime.fork_calls == []
