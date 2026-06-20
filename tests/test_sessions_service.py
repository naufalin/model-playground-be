from __future__ import annotations

import pytest

from playground.db.connection import Database
from playground.db.models import Base, LlmModel, ModelThread, PlaygroundSession, User
from playground.db.repos.thread_repo import ThreadRepo
from playground.ids import encode
from playground.sessions.service import (
    ModelNotFoundError,
    PlaygroundNotFoundError,
    PlaygroundService,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.created: list[tuple[str, str]] = []

    async def create_session(self, provider: str, model_name: str) -> str:
        self.created.append((provider, model_name))
        return f"runtime-{provider}-{model_name}"

    async def chat_stream(self, session_id: str, message: str):
        yield {"type": "text_delta", "delta": "hello "}
        yield {"type": "text_delta", "delta": "world"}


class ErrorRuntime(FakeRuntime):
    async def chat_stream(self, session_id: str, message: str):
        raise RuntimeError("runtime failed")
        yield


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
        [(model.provider, model.model_name)],
    )

    async with db.session() as session:
        threads = await ThreadRepo(session).get_by_session(playground.id)

    assert runtime.created == [("openai", "gpt-test")]
    assert len(threads) == 1
    assert threads[0].runtime_session_id == "runtime-openai-gpt-test"

    with pytest.raises(ModelNotFoundError):
        await service.stream_multi_chat(
            encode(playground.id),
            user.id,
            "hello",
            [("openai", "missing")],
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
        [(model.provider, model.model_name)],
    )
    chunks = [chunk async for chunk in stream]

    async with db.session() as session:
        threads = await ThreadRepo(session).get_by_session(playground.id)
        messages = threads[0].messages

    assert chunks[-1] == 'data: {"type": "all_done"}\n\n'
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "hello"
    assert messages[1].content == "hello world"


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
