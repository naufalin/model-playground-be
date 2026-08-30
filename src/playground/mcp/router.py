"""Authenticated proxy endpoints for the runtime MCP catalog."""

from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from playground.auth.deps import get_current_user
from playground.db.models import User
from playground.deps import get_runtime_client
from playground.mcp.schemas import McpServerListResponse, McpServerToolsResponse
from playground.runtime.client import AgentRuntimeClient

router = APIRouter(prefix="/mcp", tags=["mcp"])


def _runtime_error(exc: httpx.HTTPStatusError | httpx.RequestError) -> HTTPException:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 404):
        try:
            payload = exc.response.json()
        except ValueError:
            payload = None
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, str) and detail:
            return HTTPException(status_code=exc.response.status_code, detail=detail)
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Could not reach agent runtime MCP catalog",
    )


@router.get("/servers", response_model=McpServerListResponse)
async def list_mcp_servers(
    _user: User = Depends(get_current_user),
    runtime: AgentRuntimeClient = Depends(get_runtime_client),
) -> dict[str, Any]:
    try:
        return await runtime.list_mcp_servers()
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        raise _runtime_error(exc) from exc


@router.get("/servers/{server_id}/tools", response_model=McpServerToolsResponse)
async def list_mcp_server_tools(
    server_id: str,
    _user: User = Depends(get_current_user),
    runtime: AgentRuntimeClient = Depends(get_runtime_client),
) -> dict[str, Any]:
    try:
        return await runtime.list_mcp_server_tools(server_id)
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        raise _runtime_error(exc) from exc
