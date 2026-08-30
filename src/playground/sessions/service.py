from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from playground.db.connection import Database
from playground.db.models import Message, ModelThread, PlaygroundSession
from playground.db.repos.model_repo import ModelRepo
from playground.db.repos.session_repo import SessionRepo
from playground.db.repos.thread_repo import ThreadRepo
from playground.ids import decode, encode
from playground.mcp.catalog import APPROVED_MCP_SERVER_IDS, MCP_TOOL_PREFIX, mcp_tool_names
from playground.runtime.client import AgentRuntimeClient
from playground.sessions.chat_capture import (
    CapturedMessage,
    ChatThreadCapture,
    ChatThreadInfo,
    _request_options,
)
from playground.sessions.fanout import fanout_chat
from playground.sessions.schemas import (
    MessageOut,
    PlaygroundDetail,
    PlaygroundListOut,
    PlaygroundMode,
    PlaygroundOut,
    ThreadOut,
)

ModelSelection = tuple[str, str, str | None]


@dataclass(frozen=True)
class PreparedModelSelection:
    provider: str
    model_name: str
    reasoning_effort: str | None
    model_id: int | None
    thread: ModelThread | None


@dataclass(frozen=True)
class RegenerationPreparation:
    source_runtime_session_id: str
    prior_user_turns: int
    reasoning_effort: str | None
    mcp_servers: list[str]
    mcp_tools: list[str]


@dataclass(frozen=True)
class McpUpdateProjection:
    """MCP-relevant before/after values for an optimistic config update."""

    current_servers: frozenset[str]
    current_tools: frozenset[str]
    next_servers: list[str]
    next_server_set: frozenset[str]
    next_tools: list[str] | None
    next_orchestration: dict | None

    @property
    def next_tools_set(self) -> frozenset[str]:
        return frozenset(_selected_mcp_tools(self.next_tools, self.next_orchestration))

    @property
    def configuration_changed(self) -> bool:
        return (
            self.next_server_set != self.current_servers
            or len(self.next_servers) != len(self.next_server_set)
            or self.next_tools_set != self.current_tools
        )


class PlaygroundError(Exception):
    status_code = 400

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class PlaygroundNotFoundError(PlaygroundError):
    status_code = 404


class ModelNotFoundError(PlaygroundError):
    status_code = 400


class MessageNotFoundError(PlaygroundError):
    status_code = 404


class ThreadChangedError(PlaygroundError):
    status_code = 409


class PlaygroundChangedError(PlaygroundError):
    status_code = 409


class SystemPromptLockedError(PlaygroundError):
    status_code = 409


class RuntimeForkError(PlaygroundError):
    status_code = 502


class McpConfigurationError(PlaygroundError):
    """The requested MCP IDs/tools are not an approved runtime selection."""


class McpDiscoveryError(PlaygroundError):
    """The runtime MCP catalog could not be reached while validating config."""

    status_code = 502


def _mcp_snapshot_values(thread: ModelThread) -> tuple[list[str], list[str]]:
    """Read thread MCP snapshots while tolerating pre-MCP legacy objects."""

    return list(thread.mcp_servers_json or []), list(thread.mcp_tools_json or [])


def _selected_mcp_tools(
    tools: list[str] | None,
    orchestration: dict | None = None,
) -> list[str]:
    """Collect main and specialist MCP names from a saved configuration."""

    selected = mcp_tool_names(tools)
    for specialist in (orchestration or {}).get("specialists", []):
        if isinstance(specialist, dict):
            selected.extend(mcp_tool_names(specialist.get("tools")))
    return selected


def _mcp_update_projection(
    playground: PlaygroundSession,
    *,
    tools: list[str] | None,
    update_tools: bool,
    mcp_servers: list[str] | None,
    update_mcp_servers: bool,
    orchestration,
    update_orchestration: bool,
) -> McpUpdateProjection:
    current_servers = list(playground.mcp_servers_json or [])
    next_servers = (
        list(mcp_servers)
        if update_mcp_servers and mcp_servers is not None
        else current_servers
    )
    next_tools = tools if update_tools else playground.tools_json
    if update_orchestration:
        next_orchestration = (
            orchestration.model_dump() if orchestration is not None else None
        )
    else:
        next_orchestration = playground.orchestration_json
    return McpUpdateProjection(
        current_servers=frozenset(current_servers),
        current_tools=frozenset(
            _selected_mcp_tools(playground.tools_json, playground.orchestration_json)
        ),
        next_servers=next_servers,
        next_server_set=frozenset(next_servers),
        next_tools=next_tools,
        next_orchestration=next_orchestration,
    )


def _thread_request_options(
    thread: ModelThread,
    reasoning_effort: str | None = None,
) -> dict:
    """Build message metadata from the thread's immutable configuration."""

    mcp_servers, mcp_tools = _mcp_snapshot_values(thread)
    return _request_options(
        thread.provider,
        thread.model_name,
        reasoning_effort,
        mcp_servers=mcp_servers if mcp_servers or mcp_tools else None,
        mcp_tools=mcp_tools if mcp_servers or mcp_tools else None,
    )


def _effective_thread_tools(
    requested_tools: list[str] | None,
    thread: ModelThread,
    *,
    reject_outside_mcp: bool = True,
) -> list[str] | None:
    """Apply a per-turn tool list without allowing MCP snapshot expansion.

    ``None`` delegates the saved session selection to the runtime and ``[]``
    explicitly disables every tool.  For a non-empty explicit list, the
    immutable MCP tools captured on the thread are retained.  Callers may
    reject or discard newly requested MCP names depending on whether the
    request targets one thread or a mixed model comparison.
    """

    if requested_tools is None or requested_tools == []:
        return requested_tools

    _servers, snapshot_tools = _mcp_snapshot_values(thread)
    requested_mcp = mcp_tool_names(requested_tools)
    outside_snapshot = [name for name in requested_mcp if name not in snapshot_tools]
    if outside_snapshot and reject_outside_mcp:
        raise McpConfigurationError(
            "MCP tool(s) are not enabled on this thread: "
            f"{outside_snapshot}"
        )

    effective = [
        name
        for name in requested_tools
        if not name.startswith(MCP_TOOL_PREFIX) or name in snapshot_tools
    ]
    for name in snapshot_tools:
        if name not in effective:
            effective.append(name)
    return effective


def _decode_id(encoded_id: str, detail: str) -> int:
    try:
        return decode(encoded_id)
    except ValueError as exc:
        raise PlaygroundNotFoundError(detail) from exc


class PlaygroundService:
    def __init__(self, db: Database, runtime: AgentRuntimeClient) -> None:
        self.db = db
        self.runtime = runtime

    async def _validate_mcp_configuration(
        self,
        mcp_servers: list[str],
        tools: list[str] | None,
        orchestration: dict | None = None,
    ) -> None:
        """Validate product IDs and selected MCP tools against runtime discovery."""

        selected_servers = list(mcp_servers)
        if len(selected_servers) != len(set(selected_servers)):
            raise McpConfigurationError("MCP server IDs must be unique")

        unknown_servers = [
            server_id
            for server_id in selected_servers
            if server_id not in APPROVED_MCP_SERVER_IDS
        ]
        if unknown_servers:
            raise McpConfigurationError(f"Unknown MCP server(s): {unknown_servers}")

        selected_tools = _selected_mcp_tools(tools, orchestration)
        if selected_tools and not selected_servers:
            raise McpConfigurationError(
                "MCP tools require at least one enabled MCP server"
            )
        if not selected_servers:
            return

        list_servers = getattr(self.runtime, "list_mcp_servers", None)
        list_server_tools = getattr(self.runtime, "list_mcp_server_tools", None)
        if not callable(list_servers) or (selected_tools and not callable(list_server_tools)):
            raise McpDiscoveryError("Agent runtime does not support MCP discovery")

        try:
            server_payload = await list_servers()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            raise McpDiscoveryError("Could not reach agent runtime MCP catalog") from exc
        except Exception as exc:
            raise McpDiscoveryError("Could not read agent runtime MCP catalog") from exc
        if not isinstance(server_payload, dict):
            raise McpDiscoveryError("Agent runtime returned an invalid MCP catalog")

        discovered_servers = {
            item.get("id")
            for item in server_payload.get("servers", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        unavailable_servers = [
            server_id for server_id in selected_servers if server_id not in discovered_servers
        ]
        if unavailable_servers:
            raise McpConfigurationError(
                f"MCP server(s) are unavailable: {unavailable_servers}"
            )

        if not selected_tools:
            return

        discovered_tools: set[str] = set()
        servers_with_selected_tools = [
            server_id
            for server_id in selected_servers
            if any(
                name.startswith(f"{MCP_TOOL_PREFIX}{server_id}__")
                for name in selected_tools
            )
        ]
        for server_id in servers_with_selected_tools:
            try:
                tool_payload = await list_server_tools(server_id)
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                raise McpDiscoveryError(
                    f"Could not discover MCP tools for server '{server_id}'"
                ) from exc
            except Exception as exc:
                raise McpDiscoveryError(
                    f"Could not read MCP tools for server '{server_id}'"
                ) from exc
            if not isinstance(tool_payload, dict):
                raise McpDiscoveryError(
                    f"Agent runtime returned an invalid MCP response for '{server_id}'"
                )
            if tool_payload.get("status") != "available":
                continue
            for item in tool_payload.get("tools", []):
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if isinstance(name, str) and name.startswith(MCP_TOOL_PREFIX):
                    discovered_tools.add(name)

        unknown_tools = [name for name in selected_tools if name not in discovered_tools]
        if unknown_tools:
            raise McpConfigurationError(f"Unknown MCP tool(s): {unknown_tools}")

    async def create_playground(
        self,
        user_id: int,
        title: str,
        mode: PlaygroundMode = "compare",
        tools: list[str] | None = None,
        mcp_servers: list[str] | None = None,
        skills: list[str] | None = None,
        orchestration=None,
        system_prompt_name: str | None = None,
        system_prompt_content: str | None = None,
    ) -> PlaygroundOut:
        configured_mcp_servers = list(mcp_servers or [])
        await self._validate_mcp_configuration(
            configured_mcp_servers,
            tools,
            orchestration.model_dump() if orchestration else None,
        )
        if system_prompt_name is None or system_prompt_content is None:
            system_prompt_name, system_prompt_content = await self._get_default_system_prompt()
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            playground = await session_repo.create(
                user_id=user_id,
                title=title,
                mode=mode,
                tools=tools,
                mcp_servers=configured_mcp_servers,
                skills=skills,
                orchestration=orchestration.model_dump() if orchestration else None,
                system_prompt_name=system_prompt_name,
                system_prompt_content=system_prompt_content,
            )
            return PlaygroundOut(
                id=encode(playground.id),
                title=playground.title,
                mode=playground.mode,
                tools=playground.tools_json,
                mcp_servers=list(playground.mcp_servers_json or []),
                skills=playground.skills_json,
                orchestration=playground.orchestration_json,
                system_prompt_name=playground.system_prompt_name,
                system_prompt_content=playground.system_prompt_content,
                created_at=playground.created_at,
            )

    async def _get_default_system_prompt(self) -> tuple[str, str]:
        payload = await self.runtime.list_prompts()
        prompts = payload.get("prompts", [])
        default = next(
            (prompt for prompt in prompts if prompt.get("name") == "default"),
            prompts[0] if prompts else None,
        )
        if not default or not default.get("content"):
            raise PlaygroundError("No default system prompt is available")
        return "Default", str(default["content"])

    async def _ensure_system_prompt_snapshot(
        self,
        session_id: int,
        user_id: int,
    ) -> None:
        async with self.db.session() as session:
            repo = SessionRepo(session)
            playground = await repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            if playground.system_prompt_content is not None:
                return

        name, content = await self._get_default_system_prompt()
        async with self.db.session() as session:
            repo = SessionRepo(session)
            playground = await repo.get_if_owner_for_update(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            if playground.system_prompt_content is None:
                await repo.update_system_prompt(session_id, user_id, name, content)

    async def list_playgrounds(self, user_id: int, limit: int, offset: int) -> PlaygroundListOut:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            sessions = await session_repo.list_by_user(user_id=user_id, limit=limit, offset=offset)
            total = await session_repo.count_by_user(user_id)
            items = [
                PlaygroundOut(
                    id=encode(s.id),
                    title=s.title,
                    mode=s.mode,
                    tools=s.tools_json,
                    mcp_servers=list(s.mcp_servers_json or []),
                    skills=s.skills_json,
                    orchestration=s.orchestration_json,
                    system_prompt_name=s.system_prompt_name,
                    created_at=s.created_at,
                )
                for s in sessions
            ]
            return PlaygroundListOut(sessions=items, total=total)

    async def get_playground(self, encoded_id: str, user_id: int) -> PlaygroundDetail:
        session_id = _decode_id(encoded_id, "Playground not found")
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)
            model_repo = ModelRepo(session)

            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")

            threads = await thread_repo.get_by_session(session_id)
            thread_outs: list[ThreadOut] = []
            for thread in threads:
                await thread_repo.ensure_mcp_snapshot(thread)
                mcp_servers, mcp_tools = _mcp_snapshot_values(thread)
                model = await model_repo.get_by_provider_model(thread.provider, thread.model_name)
                display_name = model.display_name if model else thread.model_name
                # Sort messages by creation time — selectinload doesn't guarantee order
                sorted_messages = sorted(thread.messages, key=lambda m: (m.created_at, m.id))
                messages = [
                    MessageOut(
                        id=message.id,
                        role=message.role,
                        content=message.content,
                        latency_ms=message.latency_ms,
                        provider=message.provider,
                        model=message.model,
                        usage=message.usage_json,
                        thinking=message.thinking_json,
                        tool_name=message.tool_name,
                        tool_call_id=message.tool_call_id,
                        tool_input=message.tool_input,
                        output_preview=message.output_preview,
                        viz_html=message.viz_html,
                        output_delta_count=message.output_delta_count,
                        selected_skill=message.selected_skill,
                        turn_id=message.turn_id,
                        transcript_sequence=message.transcript_sequence,
                        request_options=message.request_options_json,
                        created_at=message.created_at,
                    )
                    for message in sorted_messages
                ]
                thread_outs.append(
                    ThreadOut(
                        id=encode(thread.id),
                        provider=thread.provider,
                        model_name=thread.model_name,
                        display_name=display_name,
                        mcp_servers=mcp_servers,
                        mcp_tools=mcp_tools,
                        messages=messages,
                    )
                )

            return PlaygroundDetail(
                id=encode(playground.id),
                title=playground.title,
                mode=playground.mode,
                tools=playground.tools_json,
                mcp_servers=list(playground.mcp_servers_json or []),
                skills=playground.skills_json,
                orchestration=playground.orchestration_json,
                system_prompt_name=playground.system_prompt_name,
                system_prompt_content=playground.system_prompt_content,
                created_at=playground.created_at,
                threads=thread_outs,
            )

    async def update_playground(
        self,
        encoded_id: str,
        user_id: int,
        title: str | None = None,
        tools: list[str] | None = None,
        update_tools: bool = False,
        mcp_servers: list[str] | None = None,
        update_mcp_servers: bool = False,
        skills: list[str] | None = None,
        update_skills: bool = False,
        orchestration=None,
        update_orchestration: bool = False,
        system_prompt_name: str | None = None,
        system_prompt_content: str | None = None,
        update_system_prompt: bool = False,
    ) -> PlaygroundOut:
        session_id = _decode_id(encoded_id, "Playground not found")
        if update_mcp_servers and mcp_servers is None:
            raise McpConfigurationError("mcp_servers must be a list")

        mcp_projection: McpUpdateProjection | None = None
        if update_mcp_servers or update_tools or update_orchestration:
            # Read the proposed configuration before doing discovery.  The
            # runtime call must not hold the playground row lock open.
            async with self.db.session() as session:
                session_repo = SessionRepo(session)
                playground = await session_repo.get_if_owner(session_id, user_id)
                if playground is None:
                    raise PlaygroundNotFoundError("Playground not found")
                if update_orchestration:
                    if playground.mode != "single":
                        raise PlaygroundError(
                            "Orchestration is currently available only in single mode"
                        )
                    if playground.comparison_started_at is not None:
                        raise PlaygroundError(
                            "Specialist configuration cannot be changed after a playground starts"
                        )
                mcp_projection = _mcp_update_projection(
                    playground,
                    tools=tools,
                    update_tools=update_tools,
                    mcp_servers=mcp_servers,
                    update_mcp_servers=update_mcp_servers,
                    orchestration=orchestration,
                    update_orchestration=update_orchestration,
                )

            if mcp_projection.configuration_changed:
                await self._validate_mcp_configuration(
                    mcp_projection.next_servers,
                    mcp_projection.next_tools,
                    mcp_projection.next_orchestration,
                )

        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            playground = await session_repo.get_if_owner_for_update(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            if mcp_projection is not None:
                locked_projection = _mcp_update_projection(
                    playground,
                    tools=tools,
                    update_tools=update_tools,
                    mcp_servers=mcp_servers,
                    update_mcp_servers=update_mcp_servers,
                    orchestration=orchestration,
                    update_orchestration=update_orchestration,
                )
                if (
                    locked_projection.next_server_set != mcp_projection.next_server_set
                    or locked_projection.next_tools_set != mcp_projection.next_tools_set
                ):
                    raise PlaygroundChangedError(
                        "Playground configuration changed; retry the update"
                    )

            if update_orchestration:
                if playground.mode != "single":
                    raise PlaygroundError(
                        "Orchestration is currently available only in single mode"
                    )
                if playground.comparison_started_at is not None:
                    raise PlaygroundError(
                        "Specialist configuration cannot be changed after a playground starts"
                    )

            if title is not None:
                playground = await session_repo.update_title(session_id, user_id, title)
            if update_tools:
                playground = await session_repo.update_tools(session_id, user_id, tools)
            if update_mcp_servers:
                playground = await session_repo.update_mcp_servers(
                    session_id,
                    user_id,
                    mcp_servers,
                )
            if update_skills:
                playground = await session_repo.update_skills(session_id, user_id, skills)
            if update_orchestration:
                playground = await session_repo.update_orchestration(
                    session_id,
                    user_id,
                    orchestration.model_dump() if orchestration else None,
                )
            if update_system_prompt:
                if system_prompt_name is None or system_prompt_content is None:
                    raise PlaygroundError(
                        "System prompt name and content must be provided together"
                    )
                if playground.comparison_started_at is not None:
                    raise SystemPromptLockedError(
                        "System prompt cannot be changed after a comparison has started"
                    )
                playground = await session_repo.update_system_prompt(
                    session_id,
                    user_id,
                    system_prompt_name,
                    system_prompt_content,
                )
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            return PlaygroundOut(
                id=encode(playground.id),
                title=playground.title,
                mode=playground.mode,
                tools=playground.tools_json,
                mcp_servers=list(playground.mcp_servers_json or []),
                skills=playground.skills_json,
                orchestration=playground.orchestration_json,
                system_prompt_name=playground.system_prompt_name,
                system_prompt_content=playground.system_prompt_content,
                created_at=playground.created_at,
            )

    async def delete_playground(self, encoded_id: str, user_id: int) -> None:
        session_id = _decode_id(encoded_id, "Playground not found")
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            await session_repo.delete(session_id)

    async def stream_multi_chat(
        self,
        encoded_id: str,
        user_id: int,
        message: str,
        models: list[ModelSelection],
        tools: list[str] | None = None,
        skills: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        session_id = _decode_id(encoded_id, "Playground not found")
        await self._ensure_system_prompt_snapshot(session_id, user_id)
        threads = await self._prepare_multi_chat(
            session_id, user_id, message, models, tools, skills
        )
        thread_tools = {
            thread.id: _effective_thread_tools(tools, thread, reject_outside_mcp=False)
            for thread, _reasoning_effort in threads
        }

        async def _stream() -> AsyncGenerator[str, None]:
            captures = {
                encode(thread.id): ChatThreadCapture(
                    ChatThreadInfo(
                        id=thread.id,
                        encoded_id=encode(thread.id),
                        provider=thread.provider,
                        model_name=thread.model_name,
                        request_options=_thread_request_options(thread, reasoning_effort),
                    )
                )
                for thread, reasoning_effort in threads
            }

            try:
                async for event in fanout_chat(
                    self.runtime,
                    threads,
                    message,
                    tools,
                    skills,
                    tools_by_thread=thread_tools,
                ):
                    capture = captures.get(event.get("thread_id"))
                    if capture is None or event.get("type") == "thread_start":
                        yield _sse(event)
                        continue

                    observed = capture.observe(event)
                    if event.get("type") == "done":
                        yield _sse(capture.thread_done_event())
                    elif observed is not None:
                        yield _sse(observed)
            except (GeneratorExit, asyncio.CancelledError):
                for capture in captures.values():
                    capture.cancel()
                raise
            finally:
                async with self.db.session() as session:
                    thread_repo = ThreadRepo(session)
                    for capture in captures.values():
                        await _persist_captured_messages(thread_repo, capture.thread.id, capture)

            yield _sse({"type": "all_done"})

        return _stream()

    async def stream_single_chat(
        self,
        encoded_id: str,
        thread_encoded_id: str,
        user_id: int,
        message: str,
        tools: list[str] | None = None,
        skills: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        session_id = _decode_id(encoded_id, "Playground not found")
        thread_id = _decode_id(thread_encoded_id, "Thread not found")
        thread = await self._prepare_single_chat(
            session_id,
            thread_id,
            user_id,
            message,
            tools=tools,
        )
        effective_tools = _effective_thread_tools(tools, thread)
        return self._stream_thread_chat(thread, message, effective_tools, skills)

    async def stream_regenerated_chat(
        self,
        encoded_id: str,
        thread_encoded_id: str,
        message_id: int,
        user_id: int,
        message: str,
        tools: list[str] | None = None,
        skills: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        session_id = _decode_id(encoded_id, "Playground not found")
        thread_id = _decode_id(thread_encoded_id, "Thread not found")
        preparation = await self._prepare_regeneration(
            session_id,
            thread_id,
            message_id,
            user_id,
            tools=tools,
        )

        try:
            forked_runtime_session_id = await self.runtime.fork_session(
                preparation.source_runtime_session_id,
                preparation.prior_user_turns,
            )
        except Exception as exc:
            raise RuntimeForkError("Could not prepare message regeneration") from exc

        thread = await self._replace_regeneration_tail(
            session_id,
            thread_id,
            message_id,
            user_id,
            preparation,
            forked_runtime_session_id,
            message,
        )
        return self._stream_thread_chat(
            thread,
            message,
            _effective_thread_tools(tools, thread),
            skills,
            reasoning_effort=preparation.reasoning_effort,
        )

    def _stream_thread_chat(
        self,
        thread: ModelThread,
        message: str,
        tools: list[str] | None,
        skills: list[str] | None,
        *,
        reasoning_effort: str | None = None,
    ) -> AsyncGenerator[str, None]:

        async def _stream() -> AsyncGenerator[str, None]:
            thread_id_enc = encode(thread.id)
            capture = ChatThreadCapture(
                ChatThreadInfo(
                    id=thread.id,
                    encoded_id=thread_id_enc,
                    provider=thread.provider,
                    model_name=thread.model_name,
                    request_options=_thread_request_options(thread, reasoning_effort),
                )
            )
            yield _sse(capture.start_event())
            failed = False
            try:
                async for event in self.runtime.chat_stream(
                    thread.runtime_session_id,
                    message,
                    provider=thread.provider,
                    model=thread.model_name,
                    reasoning_effort=reasoning_effort,
                    tools=tools,
                    skills=skills,
                ):
                    event["thread_id"] = thread_id_enc
                    if event.get("type") == "error":
                        failed = True
                        capture.observe(event)
                        yield _sse(event)
                        break
                    observed = capture.observe(event)
                    if observed is not None:
                        yield _sse(observed)
            except Exception as exc:
                failed = True
                yield _sse(capture.error_event(exc))
            except (GeneratorExit, asyncio.CancelledError):
                capture.cancel()
                raise
            finally:
                async with self.db.session() as session:
                    thread_repo = ThreadRepo(session)
                    await _persist_captured_messages(thread_repo, thread.id, capture)

            if not failed:
                yield _sse(capture.thread_done_event())
            yield _sse({"type": "all_done"})

        return _stream()

    async def _prepare_multi_chat(
        self,
        session_id: int,
        user_id: int,
        message: str,
        models: list[ModelSelection],
        tools: list[str] | None = None,
        skills: list[str] | None = None,
    ) -> list[tuple[ModelThread, str | None]]:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)
            model_repo = ModelRepo(session)

            playground = await session_repo.get_if_owner_for_update(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            if playground.mode == "single" and len(models) != 1:
                raise PlaygroundError("Single-mode playgrounds require exactly one model")
            if playground.mode == "single":
                existing_threads = await thread_repo.get_by_session(session_id)
                requested_model = models[0][:2]
                if existing_threads and any(
                    (thread.provider, thread.model_name) != requested_model
                    for thread in existing_threads
                ):
                    raise PlaygroundError(
                        "The model is locked after a Single-mode playground starts"
                    )
            await thread_repo.ensure_mcp_snapshot_for_session(session_id)
            playground.comparison_started_at = playground.comparison_started_at or datetime.now(UTC)
            prompt_content = playground.system_prompt_content
            mcp_servers = list(playground.mcp_servers_json or [])
            creation_tools = tools if tools is not None else playground.tools_json
            await session.flush()

            prepared: list[PreparedModelSelection] = []
            for provider, model_name, reasoning_effort in models:
                model = await model_repo.get_by_provider_model(provider, model_name)
                if model is None:
                    raise ModelNotFoundError(f"Model not found: {provider}/{model_name}")

                thread = await thread_repo.get_by_session_and_model(
                    session_id,
                    provider,
                    model_name,
                )
                if thread is not None:
                    await thread_repo.ensure_mcp_snapshot(thread)
                prepared.append(
                    PreparedModelSelection(
                        provider=provider,
                        model_name=model_name,
                        reasoning_effort=reasoning_effort,
                        model_id=model.id,
                        thread=thread,
                    )
                )

        runtime_session_ids: dict[tuple[str, str], str] = {}
        for item in prepared:
            if item.thread is not None:
                continue
            key = (item.provider, item.model_name)
            if key not in runtime_session_ids:
                create_options = {
                    "title": f"{item.provider}/{item.model_name}",
                    "tools": creation_tools,
                    "skills": skills,
                    "system_prompt": prompt_content,
                }
                if mcp_servers:
                    create_options["mcp_servers"] = mcp_servers
                if playground.orchestration_json is not None:
                    create_options["orchestration"] = playground.orchestration_json
                runtime_session_ids[key] = await self.runtime.create_session(**create_options)

        async with self.db.session() as session:
            thread_repo = ThreadRepo(session)
            threads: list[tuple[ModelThread, str | None]] = []

            for item in prepared:
                thread = item.thread
                if thread is None:
                    thread = await thread_repo.get_by_session_and_model(
                        session_id,
                        item.provider,
                        item.model_name,
                    )
                if thread is None:
                    thread = await thread_repo.create(
                        playground_session_id=session_id,
                        provider=item.provider,
                        model_name=item.model_name,
                        runtime_session_id=runtime_session_ids[(item.provider, item.model_name)],
                        model_id=item.model_id,
                        mcp_servers=mcp_servers,
                        mcp_tools=mcp_tool_names(creation_tools),
                    )
                else:
                    await thread_repo.ensure_mcp_snapshot(thread)
                threads.append((thread, item.reasoning_effort))

            for thread, reasoning_effort in threads:
                await thread_repo.add_message(
                    thread.id,
                    role="user",
                    content=message,
                    request_options_json=_thread_request_options(thread, reasoning_effort),
                )

            return threads

    async def _prepare_single_chat(
        self,
        session_id: int,
        thread_id: int,
        user_id: int,
        message: str,
        tools: list[str] | None = None,
    ) -> ModelThread:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)

            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")

            thread = await thread_repo.get(thread_id)
            if thread is None or thread.playground_session_id != session_id:
                raise PlaygroundNotFoundError("Thread not found")

            await thread_repo.ensure_mcp_snapshot(thread)
            _effective_thread_tools(tools, thread)
            await thread_repo.add_message(
                thread_id,
                role="user",
                content=message,
                request_options_json=_thread_request_options(thread),
            )
            return thread

    async def _prepare_regeneration(
        self,
        session_id: int,
        thread_id: int,
        message_id: int,
        user_id: int,
        tools: list[str] | None = None,
    ) -> RegenerationPreparation:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)

            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")

            thread = await thread_repo.get(thread_id)
            if thread is None or thread.playground_session_id != session_id:
                raise PlaygroundNotFoundError("Thread not found")

            await thread_repo.ensure_mcp_snapshot(thread)
            _effective_thread_tools(tools, thread)
            messages = sorted(thread.messages, key=lambda item: (item.created_at, item.id))
            target_index = next(
                (index for index, item in enumerate(messages) if item.id == message_id),
                None,
            )
            if target_index is None or messages[target_index].role != "user":
                raise MessageNotFoundError("User message not found")

            target = messages[target_index]
            prior_user_turns = sum(item.role == "user" for item in messages[:target_index])
            mcp_servers, mcp_tools = _mcp_snapshot_values(thread)
            return RegenerationPreparation(
                source_runtime_session_id=thread.runtime_session_id,
                prior_user_turns=prior_user_turns,
                reasoning_effort=_reasoning_effort(target),
                mcp_servers=mcp_servers,
                mcp_tools=mcp_tools,
            )

    async def _replace_regeneration_tail(
        self,
        session_id: int,
        thread_id: int,
        message_id: int,
        user_id: int,
        preparation: RegenerationPreparation,
        forked_runtime_session_id: str,
        message: str,
    ) -> ModelThread:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)

            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")

            thread = await thread_repo.get_for_update(thread_id)
            if thread is None or thread.playground_session_id != session_id:
                raise PlaygroundNotFoundError("Thread not found")
            if thread.runtime_session_id != preparation.source_runtime_session_id:
                raise ThreadChangedError("Thread changed; reload before regenerating")
            await thread_repo.ensure_mcp_snapshot(thread)
            if _mcp_snapshot_values(thread) != (
                preparation.mcp_servers,
                preparation.mcp_tools,
            ):
                raise ThreadChangedError("Thread changed; reload before regenerating")

            target = next((item for item in thread.messages if item.id == message_id), None)
            if target is None or target.role != "user":
                raise MessageNotFoundError("User message not found")

            return await thread_repo.replace_tail_with_user_message(
                thread,
                message_id,
                forked_runtime_session_id,
                message,
                _request_options(
                    thread.provider,
                    thread.model_name,
                    preparation.reasoning_effort,
                    mcp_servers=preparation.mcp_servers
                    if preparation.mcp_servers or preparation.mcp_tools
                    else None,
                    mcp_tools=preparation.mcp_tools
                    if preparation.mcp_servers or preparation.mcp_tools
                    else None,
                ),
            )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _reasoning_effort(message: Message) -> str | None:
    options = message.request_options_json
    if not isinstance(options, dict):
        return None
    effort = options.get("reasoning_effort")
    return effort if isinstance(effort, str) and effort else None


async def _persist_captured_messages(
    thread_repo: ThreadRepo,
    thread_id: int,
    capture: ChatThreadCapture,
) -> None:
    for message in capture.captured_messages():
        await _persist_captured_message(thread_repo, thread_id, message)


async def _persist_captured_message(
    thread_repo: ThreadRepo,
    thread_id: int,
    message: CapturedMessage,
) -> None:
    await thread_repo.add_message(
        thread_id,
        role=message.role,
        content=message.content,
        latency_ms=message.latency_ms,
        tool_name=message.tool_name,
        tool_call_id=message.tool_call_id,
        tool_input=message.tool_input,
        output_preview=message.output_preview,
        viz_html=message.viz_html,
        provider=message.provider,
        model=message.model,
        usage_json=message.usage_json,
        thinking_json=message.thinking_json,
        request_options_json=message.request_options_json,
        output_delta_count=message.output_delta_count,
        selected_skill=message.selected_skill,
        turn_id=message.turn_id,
        transcript_sequence=message.transcript_sequence,
    )
