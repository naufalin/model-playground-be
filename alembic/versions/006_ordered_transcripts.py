"""Add ordered transcript metadata to messages.

Revision ID: 006_ordered_transcripts
Revises: 005_playground_skills
"""

import sqlalchemy as sa

from alembic import op

revision = "006_ordered_transcripts"
down_revision = "005_playground_skills"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("turn_id", sa.String(36), nullable=True))
    op.add_column("messages", sa.Column("transcript_sequence", sa.Integer(), nullable=True))
    op.create_index(
        "ix_messages_turn_transcript_sequence",
        "messages",
        ["thread_id", "turn_id", "transcript_sequence"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_messages_turn_transcript_sequence", table_name="messages")
    op.drop_column("messages", "transcript_sequence")
    op.drop_column("messages", "turn_id")
