"""add runtime metadata columns

Revision ID: 002_runtime_metadata
Revises: 001_initial
Create Date: 2026-06-21
"""

from alembic import op
import sqlalchemy as sa


revision = "002_runtime_metadata"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("llm_models", "model_name", type_=sa.String(200), existing_nullable=False)
    op.add_column("llm_models", sa.Column("runtime_model_id", sa.Integer(), nullable=True))
    op.add_column(
        "llm_models",
        sa.Column("supports_reasoning", sa.Boolean(), server_default=sa.text("false")),
    )
    op.add_column("llm_models", sa.Column("sort_order", sa.Integer(), server_default="0"))
    op.add_column("llm_models", sa.Column("config_json", sa.JSON(), nullable=True))
    op.add_column(
        "llm_models",
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.alter_column("llm_models", "supports_reasoning", nullable=False)
    op.alter_column("llm_models", "sort_order", nullable=False)

    op.alter_column("model_threads", "model_name", type_=sa.String(200), existing_nullable=False)

    op.add_column("messages", sa.Column("tool_name", sa.String(100), nullable=True))
    op.add_column("messages", sa.Column("tool_call_id", sa.String(100), nullable=True))
    op.add_column("messages", sa.Column("tool_input", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("output_preview", sa.String(500), nullable=True))
    op.add_column("messages", sa.Column("provider", sa.String(50), nullable=True))
    op.add_column("messages", sa.Column("model", sa.String(200), nullable=True))
    op.add_column("messages", sa.Column("usage_json", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("thinking_json", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("request_options_json", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("output_delta_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "output_delta_count")
    op.drop_column("messages", "request_options_json")
    op.drop_column("messages", "thinking_json")
    op.drop_column("messages", "usage_json")
    op.drop_column("messages", "model")
    op.drop_column("messages", "provider")
    op.drop_column("messages", "output_preview")
    op.drop_column("messages", "tool_input")
    op.drop_column("messages", "tool_call_id")
    op.drop_column("messages", "tool_name")

    op.alter_column("model_threads", "model_name", type_=sa.String(100), existing_nullable=False)

    op.drop_column("llm_models", "updated_at")
    op.drop_column("llm_models", "config_json")
    op.drop_column("llm_models", "sort_order")
    op.drop_column("llm_models", "supports_reasoning")
    op.drop_column("llm_models", "runtime_model_id")
    op.alter_column("llm_models", "model_name", type_=sa.String(100), existing_nullable=False)
