"""initial schema + seed llm_models

Revision ID: 001_initial
Revises:
Create Date: 2026-06-19
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), sa.Identity(always=False), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )

    # --- llm_models ---
    op.create_table(
        "llm_models",
        sa.Column("id", sa.Integer(), sa.Identity(always=False), primary_key=True),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("display_name", sa.String(150), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("provider", "model_name", name="uq_provider_model"),
    )

    # Seed models
    llm_models = sa.table(
        "llm_models",
        sa.column("provider", sa.String),
        sa.column("model_name", sa.String),
        sa.column("display_name", sa.String),
    )
    op.bulk_insert(
        llm_models,
        [
            {"provider": "openai", "model_name": "gpt-5.4-nano", "display_name": "GPT-5.4 Nano"},
            {"provider": "openai", "model_name": "gpt-5.4-mini", "display_name": "GPT-5.4 Mini"},
            {"provider": "openai", "model_name": "gpt-5.4", "display_name": "GPT-5.4"},
            {"provider": "anthropic", "model_name": "sonnet-4.6", "display_name": "Claude Sonnet 4.6"},
            {"provider": "anthropic", "model_name": "opus-4.6", "display_name": "Claude Opus 4.6"},
            {"provider": "qwen", "model_name": "qwen-max", "display_name": "Qwen Max"},
            {"provider": "qwen", "model_name": "qwen-plus", "display_name": "Qwen Plus"},
            {"provider": "kimi", "model_name": "kimi-k2", "display_name": "Kimi K2"},
            {"provider": "minimax", "model_name": "MiniMax-M1", "display_name": "MiniMax M1"},
            {"provider": "xiaomi", "model_name": "mimo-v2.5-pro", "display_name": "MiMo v2.5 Pro"},
        ],
    )

    # --- playground_sessions ---
    op.create_table(
        "playground_sessions",
        sa.Column("id", sa.Integer(), sa.Identity(always=False), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(255), server_default="New Playground"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )

    # --- model_threads ---
    op.create_table(
        "model_threads",
        sa.Column("id", sa.Integer(), sa.Identity(always=False), primary_key=True),
        sa.Column(
            "playground_session_id",
            sa.Integer(),
            sa.ForeignKey("playground_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "model_id",
            sa.Integer(),
            sa.ForeignKey("llm_models.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("runtime_session_id", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint(
            "playground_session_id",
            "provider",
            "model_name",
            name="uq_session_provider_model",
        ),
    )

    # --- messages ---
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), sa.Identity(always=False), primary_key=True),
        sa.Column(
            "thread_id",
            sa.Integer(),
            sa.ForeignKey("model_threads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_table("model_threads")
    op.drop_table("playground_sessions")
    op.drop_table("llm_models")
    op.drop_table("users")
