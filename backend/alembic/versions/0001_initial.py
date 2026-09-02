"""initial schema: users, saved_listings, search_history, saved_searches

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "saved_listings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("listing_id", sa.Integer, nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("url", sa.String(1000), nullable=True),
        sa.Column("property_type", sa.String(50), nullable=True),
        sa.Column("subtype", sa.String(50), nullable=True),
        sa.Column("location", sa.String(500), nullable=True),
        sa.Column("price_pkr", sa.Integer, nullable=True),
        sa.Column("area_marla", sa.Float, nullable=True),
        sa.Column("saved_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "listing_id", name="uq_user_listing"),
    )
    op.create_index("ix_saved_listings_user_id", "saved_listings", ["user_id"])

    op.create_table(
        "search_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("query_text", sa.String(1000), nullable=False),
        sa.Column("intent_json", postgresql.JSONB, nullable=False),
        sa.Column("result_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_search_history_user_id", "search_history", ["user_id"])
    op.create_index("ix_search_history_created_at", "search_history", ["created_at"])

    op.create_table(
        "saved_searches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("query_text", sa.String(1000), nullable=False),
        sa.Column("intent_json", postgresql.JSONB, nullable=False),
        sa.Column("alerts_enabled", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_saved_searches_user_id", "saved_searches", ["user_id"])


def downgrade() -> None:
    op.drop_table("saved_searches")
    op.drop_table("search_history")
    op.drop_table("saved_listings")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
