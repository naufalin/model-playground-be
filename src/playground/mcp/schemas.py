"""Public MCP catalog response models."""

from typing import Literal

from pydantic import BaseModel, Field


class McpServerOut(BaseModel):
    id: str
    name: str
    description: str
    url: str
    transport: Literal["streamable_http"]
    auth_required: bool


class McpServerListResponse(BaseModel):
    servers: list[McpServerOut] = Field(default_factory=list)
    total: int


class McpToolOut(BaseModel):
    name: str
    description: str
    source: Literal["mcp"]
    server_id: str
    remote_name: str


class McpServerToolsResponse(BaseModel):
    server_id: str
    status: Literal["available", "unavailable"]
    tools: list[McpToolOut] = Field(default_factory=list)
    error: str | None = None
