"""FastAPI dependency injection for database and repos."""

from fastapi import Request

from playground.config import Settings
from playground.db.connection import Database
from playground.db.repos.model_repo import ModelRepo
from playground.db.repos.session_repo import SessionRepo
from playground.db.repos.thread_repo import ThreadRepo
from playground.db.repos.user_repo import UserRepo
from playground.runtime.client import AgentRuntimeClient


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_db(request: Request) -> Database:
    return request.app.state.db


async def get_runtime_client(request: Request) -> AgentRuntimeClient:
    return request.app.state.runtime_client


async def get_user_repo(request: Request) -> UserRepo:
    db = await get_db(request)
    return UserRepo(db)


async def get_model_repo(request: Request) -> ModelRepo:
    db = await get_db(request)
    return ModelRepo(db)


async def get_session_repo(request: Request) -> SessionRepo:
    db = await get_db(request)
    return SessionRepo(db)


async def get_thread_repo(request: Request) -> ThreadRepo:
    db = await get_db(request)
    return ThreadRepo(db)
