from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlaygroundCreate(BaseModel):
    title: str = Field(default="New Playground", min_length=1, max_length=255)


class PlaygroundUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class PlaygroundOut(BaseModel):
    id: str
    title: str
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class PlaygroundListOut(BaseModel):
    sessions: list[PlaygroundOut]
    total: int


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    latency_ms: int | None = None
    provider: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_input: dict[str, Any] | None = None
    output_preview: str | None = None
    viz_html: str | None = None
    output_delta_count: int | None = None
    request_options: dict[str, Any] | None = None
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class ThreadOut(BaseModel):
    id: str
    provider: str
    model_name: str
    display_name: str
    messages: list[MessageOut] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class PlaygroundDetail(BaseModel):
    id: str
    title: str
    created_at: datetime | None = None
    threads: list[ThreadOut] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ModelSelect(BaseModel):
    provider: str
    model_name: str
    reasoning_effort: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    models: list[ModelSelect]


class ContinueChatRequest(BaseModel):
    message: str = Field(min_length=1)
