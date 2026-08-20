"""Reproducibility fields on runs (T3.5.12/D12) and statistical validity on hypotheses (D8).

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-20

These five columns were added to `models.py` during slices 8 and 9 and existed only because
`Base.metadata.create_all` happens to add missing *tables* - it does **not** add missing
columns to a table that already exists. Any database created before those slices therefore
had the ORM mapping a `seed` column that the file did not contain, and would raise
`no such column: experiment_runs.seed` on the first query. That is the concrete reason this
migration is not bookkeeping.

All five are nullable, and that is a scientific statement rather than a convenience: rows
written before this revision genuinely have no seed and no q-value, and a fabricated default
(0, or 1.0) would be indistinguishable from a real measurement. `NULL` reads as "this run
predates seed capture", which is the truth.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: (table, column, type) for the five columns this revision adds.
ADDED = (
    ("experiment_runs", "seed", sa.Integer()),
    ("experiment_runs", "execution", sa.JSON()),
    ("hypotheses", "p_value", sa.Float()),
    ("hypotheses", "q_value", sa.Float()),
    ("hypotheses", "n_tests", sa.Integer()),
    ("hypotheses", "statistics", sa.JSON()),
)


def upgrade() -> None:
    # batch_alter_table for symmetry with downgrade(); SQLite supports plain ADD COLUMN,
    # but keeping both directions in batch mode means the script is read the same way in
    # both and there is one fewer dialect-specific case to reason about.
    for table in ("experiment_runs", "hypotheses"):
        with op.batch_alter_table(table) as batch:
            for tbl, name, type_ in ADDED:
                if tbl == table:
                    batch.add_column(sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    # DROP COLUMN needs SQLite >= 3.35; batch mode falls back to copy-and-rename below
    # that, so this downgrade works on the SQLite the platform is expected to run on.
    for table in ("experiment_runs", "hypotheses"):
        with op.batch_alter_table(table) as batch:
            for tbl, name, _type in reversed(ADDED):
                if tbl == table:
                    batch.drop_column(name)
