from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlaygroundCreate(BaseModel):
    title: str = Field(default="New Playground", min_length=1, max_length=255)
    tools: list[str] | None = Field(default=None, max_length=32)
    skills: list[str] | None = Field(default=None, max_length=32)
    system_prompt_name: str | None = Field(default=None, min_length=1, max_length=100)
    system_prompt_content: str | None = Field(
        default=None, min_length=1, max_length=32_000
    )

    @model_validator(mode="after")
    def validate_system_prompt_snapshot(self) -> PlaygroundCreate:
        if (self.system_prompt_name is None) != (self.system_prompt_content is None):
            raise ValueError("System prompt name and content must be provided together")
        return self


class PlaygroundUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    tools: list[str] | None = Field(default=None, max_length=32)
    skills: list[str] | None = Field(default=None, max_length=32)
    system_prompt_name: str | None = Field(default=None, min_length=1, max_length=100)
    system_prompt_content: str | None = Field(
        default=None, min_length=1, max_length=32_000
    )

    @model_validator(mode="after")
    def validate_system_prompt_snapshot(self) -> PlaygroundUpdate:
        prompt_fields = {"system_prompt_name", "system_prompt_content"}
        if prompt_fields & self.model_fields_set and not prompt_fields <= self.model_fields_set:
            raise ValueError("System prompt name and content must be provided together")
        return self


class PlaygroundOut(BaseModel):
    id: str
    title: str
    tools: list[str] | None = None
    skills: list[str] | None = None
    system_prompt_name: str | None = None
    system_prompt_content: str | None = None
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
    selected_skill: str | None = None
    turn_id: str | None = None
    transcript_sequence: int | None = None
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
    tools: list[str] | None = None
    skills: list[str] | None = None
    system_prompt_name: str | None = None
    system_prompt_content: str | None = None
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
    tools: list[str] | None = Field(default=None, max_length=32)
    skills: list[str] | None = Field(default=None, max_length=32)


class ContinueChatRequest(BaseModel):
    message: str = Field(min_length=1)
    tools: list[str] | None = Field(default=None, max_length=32)
    skills: list[str] | None = Field(default=None, max_length=32)


class RegenerateChatRequest(BaseModel):
    message: str = Field(min_length=1)
    tools: list[str] | None = Field(default=None, max_length=32)
    skills: list[str] | None = Field(default=None, max_length=32)
