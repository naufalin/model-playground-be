"""Add playground tool defaults.

Revision ID: 004_playground_tools
Revises: 003_message_viz_html
Create Date: 2026-06-27
"""

import sqlalchemy as sa

from alembic import op

revision = "004_playground_tools"
down_revision = "003_message_viz_html"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("playground_sessions", sa.Column("tools_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("playground_sessions", "tools_json")
