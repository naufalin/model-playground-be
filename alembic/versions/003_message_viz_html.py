"""Add visualization HTML storage to messages.

Revision ID: 003_message_viz_html
Revises: 002_runtime_metadata
Create Date: 2026-06-27
"""

import sqlalchemy as sa

from alembic import op

revision = "003_message_viz_html"
down_revision = "002_runtime_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("viz_html", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "viz_html")
