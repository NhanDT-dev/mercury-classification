"""Add analytics and metrics tables

Revision ID: 003_analytics
Revises: 002_user_roles
Create Date: 2024-02-01 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '003_analytics'
down_revision = '002_user_roles'
branch_labels = None
depends_on = None

def upgrade():
    # Create analytics tables
    pass

def downgrade():
    # Drop analytics tables
    pass
