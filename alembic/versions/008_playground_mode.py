"""Persist the playground interaction mode.

Revision ID: 008_playground_mode
Revises: 007_playground_system_prompt
"""

import sqlalchemy as sa

from alembic import op

revision = "008_playground_mode"
down_revision = "007_playground_system_prompt"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "playground_sessions",
        sa.Column(
            "mode",
            sa.String(16),
            nullable=False,
            server_default="compare",
        ),
    )


def downgrade() -> None:
    op.drop_column("playground_sessions", "mode")
