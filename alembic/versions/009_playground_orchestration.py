"""add playground orchestration snapshot"""

from alembic import op
import sqlalchemy as sa

revision = "009_playground_orchestration"
down_revision = "008_playground_mode"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("playground_sessions", sa.Column("orchestration_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("playground_sessions", "orchestration_json")
