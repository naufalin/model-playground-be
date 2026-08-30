"""Persist approved MCP defaults and per-thread snapshots."""

import sqlalchemy as sa

from alembic import op

revision = "010_playground_mcp"
down_revision = "009_playground_orchestration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The server default backfills existing playgrounds/threads as empty MCP
    # configurations while keeping the columns non-null for future snapshots.
    op.add_column(
        "playground_sessions",
        sa.Column("mcp_servers_json", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "model_threads",
        sa.Column("mcp_servers_json", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "model_threads",
        sa.Column("mcp_tools_json", sa.JSON(), nullable=False, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("model_threads", "mcp_tools_json")
    op.drop_column("model_threads", "mcp_servers_json")
    op.drop_column("playground_sessions", "mcp_servers_json")
