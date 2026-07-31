"""Persist system prompt snapshots on playground sessions.

Revision ID: 007_playground_system_prompt
Revises: 006_ordered_transcripts
"""

import sqlalchemy as sa

from alembic import op

revision = "007_playground_system_prompt"
down_revision = "006_ordered_transcripts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "playground_sessions",
        sa.Column("system_prompt_name", sa.String(100), nullable=True),
    )
    op.add_column(
        "playground_sessions",
        sa.Column("system_prompt_content", sa.Text(), nullable=True),
    )
    op.add_column(
        "playground_sessions",
        sa.Column("comparison_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE playground_sessions
            SET comparison_started_at = CURRENT_TIMESTAMP
            WHERE EXISTS (
                SELECT 1
                FROM model_threads
                WHERE model_threads.playground_session_id = playground_sessions.id
            )
            """
        )
    )


def downgrade() -> None:
    op.drop_column("playground_sessions", "comparison_started_at")
    op.drop_column("playground_sessions", "system_prompt_content")
    op.drop_column("playground_sessions", "system_prompt_name")
