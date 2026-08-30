from __future__ import annotations

import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from playground.app import create_app
from playground.auth.deps import get_current_user
from playground.db.connection import Database
from playground.db.models import Base, LlmModel, User
from playground.db.repos.thread_repo import ThreadRepo
from playground.deps import get_db, get_runtime_client
from playground.ids import encode
from playground.sessions.service import (
    McpConfigurationError,
    PlaygroundService,
)

MCP_TOOL = "mcp_deepwiki__ask_question"
MCP_AWS_TOOL = "mcp_aws_knowledge__aws___search_documentation"


class McpRuntime:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.chat_tools: list[list[str] | None] = []
        self.fork_calls: list[tuple[str, int]] = []
        self.discovery_calls: list[str] = []

    async def list_prompts(self) -> dict:
        return {"prompts": [{"name": "default", "content": "Helpful."}]}

    async def list_mcp_servers(self) -> dict:
        self.discovery_calls.append("servers")
        return {
            "servers": [
                {"id": "deepwiki"},
                {"id": "aws_knowledge"},
                {"id": "microsoft_learn"},
            ],
            "total": 3,
        }

    async def list_mcp_server_tools(self, server_id: str) -> dict:
        self.discovery_calls.append(server_id)
        names = {
            "deepwiki": [MCP_TOOL],
            "aws_knowledge": [MCP_AWS_TOOL],
            "microsoft_learn": [],
        }
        return {
            "server_id": server_id,
            "status": "available",
            "tools": [
                {
                    "name": name,
                    "description": name,
                    "source": "mcp",
                    "server_id": server_id,
                    "remote_name": name.removeprefix(f"mcp_{server_id}__"),
                }
                for name in names[server_id]
            ],
            "error": None,
        }

    async def create_session(
        self,
        title: str = "New Session",
        tools: list[str] | None = None,
        skills: list[str] | None = None,
        system_prompt: str | None = None,
        orchestration: dict | None = None,
        mcp_servers: list[str] | None = None,
    ) -> str:
        self.created.append(
            {
                "title": title,
                "tools": tools,
                "skills": skills,
                "system_prompt": system_prompt,
                "orchestration": orchestration,
                "mcp_servers": mcp_servers,
            }
        )
        return f"runtime-{len(self.created)}"

    async def fork_session(self, session_id: str, keep_user_turns: int) -> str:
        self.fork_calls.append((session_id, keep_user_turns))
        return f"fork-{session_id}"

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
        yield {"type": "text_delta", "delta": "ok"}
        yield {
            "type": "done",
            "provider": provider,
            "model": model,
            "usage": {"total_tokens": 1},
            "thinking": None,
            "output_delta_count": 1,
        }


async def make_db() -> Database:
    db = Database("sqlite+aiosqlite:///:memory:")
    db.connect()
    assert db.engine is not None
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return db


async def create_user(db: Database) -> User:
    async with db.session() as session:
        user = User(email="mcp@example.com", hashed_password="hashed")
        session.add(user)
        await session.flush()
        return user


async def create_model(db: Database, model_name: str = "gpt-test") -> LlmModel:
    async with db.session() as session:
        model = LlmModel(
            provider="openai",
            model_name=model_name,
            display_name="GPT Test",
            is_active=True,
        )
        session.add(model)
        await session.flush()
        return model


async def test_mcp_proxy_requires_auth_and_returns_runtime_catalog() -> None:
    db = await make_db()
    runtime = McpRuntime()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_runtime_client] = lambda: runtime
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            unauthorized = await client.get("/mcp/servers")
            app.dependency_overrides[get_current_user] = lambda: User(
                id=1,
                email="mcp@example.com",
                hashed_password="hashed",
            )
            response = await client.get("/mcp/servers/deepwiki/tools")
    finally:
        app.dependency_overrides.clear()
        await db.disconnect()

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json()["tools"][0]["source"] == "mcp"


async def test_mcp_defaults_and_thread_snapshot_survive_playground_updates() -> None:
    db = await make_db()
    user = await create_user(db)
    model = await create_model(db)
    runtime = McpRuntime()
    service = PlaygroundService(db, runtime)

    created = await service.create_playground(
        user.id,
        "MCP lab",
        tools=["web_search", MCP_TOOL],
        mcp_servers=["deepwiki"],
    )
    assert created.mcp_servers == ["deepwiki"]

    discovery_call_count = len(runtime.discovery_calls)
    await service.update_playground(
        created.id,
        user.id,
        tools=["web_search", "calculator", MCP_TOOL],
        update_tools=True,
    )
    assert len(runtime.discovery_calls) == discovery_call_count

    first_stream = await service.stream_multi_chat(
        created.id,
        user.id,
        "first",
        [(model.provider, model.model_name, None)],
        ["web_search", MCP_TOOL],
    )
    [chunk async for chunk in first_stream]

    await service.update_playground(
        created.id,
        user.id,
        tools=["web_search", MCP_AWS_TOOL],
        update_tools=True,
        mcp_servers=["aws_knowledge"],
        update_mcp_servers=True,
    )

    old_continuation = await service.stream_single_chat(
        created.id,
        encode(1),
        user.id,
        "old thread",
        ["web_search"],
    )
    [chunk async for chunk in old_continuation]
    second_model = await create_model(db, "gpt-second")
    second_stream = await service.stream_multi_chat(
        created.id,
        user.id,
        "second",
        [
            (model.provider, model.model_name, None),
            (second_model.provider, second_model.model_name, None),
        ],
        ["web_search", MCP_AWS_TOOL],
    )
    [chunk async for chunk in second_stream]
    assert ["web_search", MCP_TOOL] in runtime.chat_tools[-2:]
    assert ["web_search", MCP_AWS_TOOL] in runtime.chat_tools[-2:]

    async with db.session() as session:
        threads = await ThreadRepo(session).get_by_session(1)

    old_thread = next(thread for thread in threads if thread.model_name == model.model_name)
    new_thread = next(thread for thread in threads if thread.model_name == second_model.model_name)
    assert old_thread.mcp_servers_json == ["deepwiki"]
    assert old_thread.mcp_tools_json == [MCP_TOOL]
    assert new_thread.mcp_servers_json == ["aws_knowledge"]
    assert new_thread.mcp_tools_json == [MCP_AWS_TOOL]
    assert runtime.created[0]["mcp_servers"] == ["deepwiki"]
    assert runtime.created[1]["mcp_servers"] == ["aws_knowledge"]

    detail = await service.get_playground(created.id, user.id)
    detail_old = next(thread for thread in detail.threads if thread.model_name == model.model_name)
    assert detail_old.mcp_servers == ["deepwiki"]
    assert detail_old.mcp_tools == [MCP_TOOL]
    await db.disconnect()


async def test_mcp_validation_includes_specialists_but_thread_snapshot_is_main_only() -> None:
    db = await make_db()
    user = await create_user(db)
    model = await create_model(db)
    runtime = McpRuntime()
    service = PlaygroundService(db, runtime)

    from playground.sessions.schemas import OrchestrationSnapshot

    created = await service.create_playground(
        user.id,
        "Orchestrated MCP lab",
        mode="single",
        tools=[MCP_TOOL],
        mcp_servers=["deepwiki", "aws_knowledge"],
        orchestration=OrchestrationSnapshot.model_validate(
            {
                "specialists": [
                    {
                        "name": "aws_researcher",
                        "description": "Search AWS documentation",
                        "instructions": "Use AWS sources.",
                        "tools": [MCP_AWS_TOOL],
                    }
                ]
            }
        ),
    )
    stream = await service.stream_multi_chat(
        created.id,
        user.id,
        "search",
        [(model.provider, model.model_name, None)],
        [MCP_TOOL],
    )
    [chunk async for chunk in stream]

    async with db.session() as session:
        thread = (await ThreadRepo(session).get_by_session(1))[0]

    assert thread.mcp_tools_json == [MCP_TOOL]
    assert runtime.created[0]["orchestration"]["specialists"][0]["tools"] == [MCP_AWS_TOOL]
    await db.disconnect()


async def test_mcp_turn_semantics_preserve_snapshot_and_respect_null_empty() -> None:
    db = await make_db()
    user = await create_user(db)
    model = await create_model(db)
    runtime = McpRuntime()
    service = PlaygroundService(db, runtime)
    created = await service.create_playground(
        user.id,
        "MCP lab",
        tools=["web_search", MCP_TOOL],
        mcp_servers=["deepwiki"],
    )
    stream = await service.stream_multi_chat(
        created.id,
        user.id,
        "first",
        [(model.provider, model.model_name, None)],
        ["web_search", MCP_TOOL],
    )
    [chunk async for chunk in stream]

    continued = await service.stream_single_chat(
        created.id,
        encode(1),
        user.id,
        "builtin plus frozen MCP",
        ["web_search"],
    )
    [chunk async for chunk in continued]
    assert runtime.chat_tools[-1] == ["web_search", MCP_TOOL]

    delegated = await service.stream_single_chat(
        created.id,
        encode(1),
        user.id,
        "delegate",
        None,
    )
    [chunk async for chunk in delegated]
    assert runtime.chat_tools[-1] is None

    disabled = await service.stream_single_chat(
        created.id,
        encode(1),
        user.id,
        "disable",
        [],
    )
    [chunk async for chunk in disabled]
    assert runtime.chat_tools[-1] == []

    with pytest.raises(McpConfigurationError, match="not enabled"):
        await service.stream_single_chat(
            created.id,
            encode(1),
            user.id,
            "outside",
            [MCP_AWS_TOOL],
        )
    await db.disconnect()


async def test_mcp_regeneration_keeps_original_snapshot_metadata_and_history() -> None:
    db = await make_db()
    user = await create_user(db)
    model = await create_model(db)
    runtime = McpRuntime()
    service = PlaygroundService(db, runtime)
    created = await service.create_playground(
        user.id,
        "MCP lab",
        tools=[MCP_TOOL],
        mcp_servers=["deepwiki"],
    )
    initial = await service.stream_multi_chat(
        created.id,
        user.id,
        "original",
        [(model.provider, model.model_name, "high")],
        [MCP_TOOL],
    )
    [chunk async for chunk in initial]

    async with db.session() as session:
        thread = (await ThreadRepo(session).get_by_session(1))[0]
        thread_id = thread.id
        message_id = next(message.id for message in thread.messages if message.role == "user")

    regenerated = await service.stream_regenerated_chat(
        created.id,
        encode(thread_id),
        message_id,
        user.id,
        "replacement",
        None,
    )
    [chunk async for chunk in regenerated]

    async with db.session() as session:
        thread = await ThreadRepo(session).get(thread_id)
        assert thread is not None
        messages = thread.messages
    assert runtime.fork_calls == [("runtime-1", 0)]
    assert thread.mcp_servers_json == ["deepwiki"]
    assert thread.mcp_tools_json == [MCP_TOOL]
    assert messages[0].content == "replacement"
    assert messages[0].request_options_json == {
        "provider": "openai",
        "model": "gpt-test",
        "reasoning_effort": "high",
        "mcp_servers": ["deepwiki"],
        "mcp_tools": [MCP_TOOL],
    }
    await db.disconnect()


async def test_runtime_client_mcp_routes_and_session_payload() -> None:
    from playground.runtime.client import AgentRuntimeClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/mcp/servers":
            return httpx.Response(200, json={"servers": [], "total": 0})
        if request.url.path == "/mcp/servers/deepwiki/tools":
            return httpx.Response(
                200,
                json={
                    "server_id": "deepwiki",
                    "status": "available",
                    "tools": [],
                    "error": None,
                },
            )
        if request.url.path == "/sessions":
            assert json.loads(request.content)["mcp_servers"] == ["deepwiki"]
            return httpx.Response(201, json={"id": "runtime-id"})
        raise AssertionError(request.url)

    runtime = AgentRuntimeClient(base_url="http://runtime")
    runtime._client = httpx.AsyncClient(
        base_url="http://runtime",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert (await runtime.list_mcp_servers())["total"] == 0
        assert (await runtime.list_mcp_server_tools("deepwiki"))["server_id"] == "deepwiki"
        assert (
            await runtime.create_session(title="MCP", mcp_servers=["deepwiki"])
        ) == "runtime-id"
    finally:
        await runtime.close()
