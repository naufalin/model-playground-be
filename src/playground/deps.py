"""FastAPI dependency injection for external app resources."""

from fastapi import Request

from playground.config import Settings
from playground.db.connection import Database
from playground.runtime.client import AgentRuntimeClient


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_db(request: Request) -> Database:
    return request.app.state.db


async def get_runtime_client(request: Request) -> AgentRuntimeClient:
    return request.app.state.runtime_client
