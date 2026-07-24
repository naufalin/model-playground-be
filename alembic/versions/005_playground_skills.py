"""Add playground skill configuration and selected skill metadata.

Revision ID: 005_playground_skills
Revises: 004_playground_tools
"""

from alembic import op
import sqlalchemy as sa

revision = "005_playground_skills"
down_revision = "004_playground_tools"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("playground_sessions", sa.Column("skills_json", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("selected_skill", sa.String(100), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "selected_skill")
    op.drop_column("playground_sessions", "skills_json")
