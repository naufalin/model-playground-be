from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PlaygroundCreate(BaseModel):
    title: str = "New Playground"


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


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    models: list[ModelSelect]


class ContinueChatRequest(BaseModel):
    message: str = Field(min_length=1)
