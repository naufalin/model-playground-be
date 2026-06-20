"""FastAPI dependency injection for database and repos."""

from playground.db.connection import Database
from playground.db.repos.model_repo import ModelRepo
from playground.db.repos.session_repo import SessionRepo
from playground.db.repos.thread_repo import ThreadRepo
from playground.db.repos.user_repo import UserRepo

_db: Database | None = None


async def get_db() -> Database:
    """Get or create the database engine."""
    global _db
    if _db is None:
        from playground.config import settings

        _db = Database(settings.database_url)
        _db.connect()
    return _db


async def get_user_repo() -> UserRepo:
    db = await get_db()
    return UserRepo(db)


async def get_model_repo() -> ModelRepo:
    db = await get_db()
    return ModelRepo(db)


async def get_session_repo() -> SessionRepo:
    db = await get_db()
    return SessionRepo(db)


async def get_thread_repo() -> ThreadRepo:
    db = await get_db()
    return ThreadRepo(db)
