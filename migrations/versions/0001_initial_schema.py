"""Initial schema: the five tables that predate migration control.

Revision ID: 0001
Revises: None
Create Date: 2026-08-20

**This is the pre-T3.5.12 schema, on purpose.** It is tempting to make the baseline
migration match `models.py` as it stands today, but that would make the baseline unusable
for the thing a baseline is *for*: adopting an existing database. `spectral_earth.db` was
built by `Base.metadata.create_all` before slices 8 and 9 added five columns. If revision
0001 already contained those columns, stamping that database as 0001 would assert the
presence of columns it does not have, and the next `upgrade head` would skip the migration
that would have added them - leaving a database that Alembic believes is current and that
the ORM cannot query. `src.database.migrate.adopt_existing` therefore probes for those
columns and stamps 0001 or 0002 accordingly, which only works if the two revisions really
do describe different schemas.

Python-side defaults (`default="PENDING"`, the `created_at` lambdas) intentionally emit no
DDL: they are applied by SQLAlchemy on insert, not by the database. Rendering them as
server defaults here would create a difference between the migrated schema and the ORM's
view of it, which `test_migrations.py::test_no_drift` would - correctly - fail on.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_experiments_id"), "experiments", ["id"], unique=False)

    op.create_table(
        "experiment_runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("experiment_id", sa.String(), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("results", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_experiment_runs_id"), "experiment_runs", ["id"], unique=False)

    op.create_table(
        "lineage_nodes",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("experiment_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["experiment_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_lineage_nodes_id"), "lineage_nodes", ["id"], unique=False)

    op.create_table(
        "lineage_edges",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("target_id", sa.String(), nullable=False),
        sa.Column("relation", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["lineage_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_id"], ["lineage_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_lineage_edges_id"), "lineage_edges", ["id"], unique=False)

    op.create_table(
        "hypotheses",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("experiment_ids", sa.JSON(), nullable=False),
        sa.Column("pattern_type", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("metrics_analyzed", sa.JSON(), nullable=False),
        sa.Column("parameters_analyzed", sa.JSON(), nullable=False),
        sa.Column("proposed_experiment_config", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_hypotheses_id"), "hypotheses", ["id"], unique=False)


def downgrade() -> None:
    # Reverse creation order so the foreign keys never dangle. On PostgreSQL the order
    # matters (a referenced table cannot be dropped); on SQLite with foreign_keys=ON it
    # matters too, which is precisely why that pragma is enabled in session.py rather
    # than left at its permissive default.
    op.drop_index(op.f("ix_hypotheses_id"), table_name="hypotheses")
    op.drop_table("hypotheses")
    op.drop_index(op.f("ix_lineage_edges_id"), table_name="lineage_edges")
    op.drop_table("lineage_edges")
    op.drop_index(op.f("ix_lineage_nodes_id"), table_name="lineage_nodes")
    op.drop_table("lineage_nodes")
    op.drop_index(op.f("ix_experiment_runs_id"), table_name="experiment_runs")
    op.drop_table("experiment_runs")
    op.drop_index(op.f("ix_experiments_id"), table_name="experiments")
    op.drop_table("experiments")
