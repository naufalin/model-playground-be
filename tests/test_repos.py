from __future__ import annotations

from playground.db.connection import Database
from playground.db.models import Base
from playground.db.repos.session_repo import SessionRepo
from playground.db.repos.thread_repo import ThreadRepo
from playground.db.repos.user_repo import UserRepo


async def make_db() -> Database:
    db = Database("sqlite+aiosqlite:///:memory:")
    db.connect()
    assert db.engine is not None
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return db


async def test_repos_share_explicit_session_and_persist_on_context_exit() -> None:
    db = await make_db()
    try:
        async with db.session() as session:
            user = await UserRepo(session).create_user(
                email="user@example.com",
                hashed_password="hashed",
            )
            playground = await SessionRepo(session).create(user_id=user.id, title="Shared")
            thread_repo = ThreadRepo(session)
            thread = await thread_repo.create(
                playground_session_id=playground.id,
                provider="openai",
                model_name="gpt-test",
                runtime_session_id="runtime-1",
            )
            await thread_repo.add_message(thread.id, role="user", content="hello")

            in_transaction = await thread_repo.get_by_session(playground.id)
            assert len(in_transaction) == 1
            assert in_transaction[0].messages[0].content == "hello"

        async with db.session() as session:
            persisted = await ThreadRepo(session).get_by_session(playground.id)

        assert len(persisted) == 1
        assert persisted[0].runtime_session_id == "runtime-1"
        assert persisted[0].messages[0].content == "hello"
    finally:
        await db.disconnect()
