"""Add user roles and permissions

Revision ID: 002_user_roles
Revises: 001_initial
Create Date: 2024-01-20 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '002_user_roles'
down_revision = '001_initial'
branch_labels = None
depends_on = None

def upgrade():
    # Add role column to users table
    pass

def downgrade():
    # Remove role column
    pass
